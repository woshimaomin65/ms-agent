# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import hashlib
import importlib
import os
import re
import traceback
from datetime import datetime
from functools import partial, wraps
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple

import json
import json5
from ms_agent.llm.utils import Message
from ms_agent.memory import Memory
from ms_agent.utils import get_fact_retrieval_prompt
from ms_agent.utils.constants import (DEFAULT_OUTPUT_DIR, DEFAULT_SEARCH_LIMIT,
                                      DEFAULT_USER, get_service_config)
from ms_agent.utils.logger import logger
from omegaconf import DictConfig, OmegaConf


class MemoryMapping:
    """内存映射类，用于管理内存条目的状态和关联"""
    memory_id: str = None  # 内存 ID
    memory: str = None  # 内存内容
    valid: bool = None  # 是否有效
    enable_idxs: List[int] = []  # 启用的消息 ID 列表
    disable_idx: int = -1  # 禁用的消息 ID

    def __init__(self, memory_id: str, value: str, enable_idxs: int
                 or List[int]):
        """初始化内存映射

        参数：
            memory_id: 内存 ID
            value: 内存内容值
            enable_idxs: 启用的消息 ID，可以是单个整数或整数列表
        """
        self.memory_id = memory_id
        self.value = value
        self.valid = True
        if isinstance(enable_idxs, int):
            enable_idxs = [enable_idxs]
        self.enable_idxs = enable_idxs

    def udpate_idxs(self, enable_idxs: int or List[int]):
        """更新启用的消息 ID 列表"""
        if isinstance(enable_idxs, int):
            enable_idxs = [enable_idxs]
        self.enable_idxs.extend(enable_idxs)

    def disable(self, disable_idx: int):
        """禁用当前内存映射"""
        self.valid = False
        self.disable_idx = disable_idx

    def try_enable(self, expired_disable_idx: int):
        """尝试重新启用内存，当禁用 ID 过期时"""
        if expired_disable_idx == self.disable_idx:
            self.valid = True
            self.disable_idx = -1

    def get(self):
        """获取内存内容值"""
        return self.value

    def to_dict(self) -> Dict:
        """将对象转换为字典表示"""
        return {
            'memory_id': self.memory_id,
            'value': self.value,
            'valid': self.valid,
            'enable_idxs': self.enable_idxs.copy(
            ),  # 返回副本以防止外部修改
            'disable_idx': self.disable_idx
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryMapping':
        """从字典创建 MemoryMapping 实例"""
        instance = cls(
            memory_id=data['memory_id'],
            value=data['value'],
            enable_idxs=data['enable_idxs'])
        instance.valid = data['valid']
        instance.disable_idx = data.get('disable_idx',
                                        -1)  # 兼容旧数据
        return instance


class DefaultMemory(Memory):
    """默认内存管理类，用于处理记忆的存储、检索和管理"""

    def __init__(self, config: DictConfig):
        """初始化默认内存

        参数：
            config: 配置对象，包含记忆相关的各项配置
        """
        super().__init__(config)
        memory_config = config.memory.default_memory
        self.user_id: Optional[str] = getattr(memory_config, 'user_id',
                                              DEFAULT_USER)  # 用户 ID
        self.agent_id: Optional[str] = getattr(memory_config, 'agent_id', None)  # 代理 ID
        self.run_id: Optional[str] = getattr(memory_config, 'run_id', None)  # 运行 ID
        self.compress: Optional[bool] = getattr(config, 'compress', True)  # 是否压缩
        self.is_retrieve: Optional[bool] = getattr(config, 'is_retrieve', True)  # 是否检索
        self.path: Optional[str] = getattr(
            memory_config, 'path',
            os.path.join(DEFAULT_OUTPUT_DIR, '.default_memory'))  # 存储路径
        self.history_mode = getattr(memory_config, 'history_mode', 'add')  # 历史模式：add 或 overwrite
        self.ignore_roles: List[str] = getattr(memory_config, 'ignore_roles',
                                               ['tool', 'system'])  # 忽略的角色列表
        self.ignore_fields: List[str] = getattr(memory_config, 'ignore_fields',
                                                ['reasoning_content'])  # 忽略的字段列表
        self.search_limit: int = getattr(memory_config, 'search_limit',
                                         DEFAULT_SEARCH_LIMIT)  # 搜索限制数量
        # 为共享使用中的线程安全添加锁
        self._lock = asyncio.Lock()
        self.memory = self._init_memory_obj()  # 初始化内存对象
        self.load_cache()  # 加载缓存

    async def init_cache_messages(self):
        """初始化缓存消息，当有缓存消息但内存快照为空时，将缓存消息添加到内存"""
        if len(self.cache_messages) and not len(self.memory_snapshot):
            for id, messages in self.cache_messages.items():
                await self.add_single(messages, msg_id=id)

    def save_cache(self):
        """
        将 self.max_msg_id、self.cache_messages 和 self.memory_snapshot 保存到 self.path/cache_messages.json
        """
        cache_file = os.path.join(self.path, 'cache_messages.json')

        # 确保目录存在
        os.makedirs(self.path, exist_ok=True)

        data = {
            'max_msg_id': self.max_msg_id,
            'cache_messages': {
                str(k): ([msg.to_dict() for msg in msg_list], _hash)
                for k, (msg_list, _hash) in self.cache_messages.items()
            },
            'memory_snapshot': [mm.to_dict() for mm in self.memory_snapshot]
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json5.dump(data, f, indent=2, ensure_ascii=False)

    def load_cache(self):
        """
        从 self.path/cache_messages.json 加载数据到 self.max_msg_id、self.cache_messages 和 self.memory_snapshot
        """
        cache_file = os.path.join(self.path, 'cache_messages.json')

        if not os.path.exists(cache_file):
            # 如果文件不存在，初始化默认值并返回
            self.max_msg_id = -1
            self.cache_messages = {}
            self.memory_snapshot = []
            return

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json5.load(f)

            self.max_msg_id = data.get('max_msg_id', -1)

            # 解析 cache_messages
            cache_messages = {}
            raw_cache_msgs = data.get('cache_messages', {})
            for k, (msg_list, timestamp) in raw_cache_msgs.items():
                msg_objs = [Message(**msg_dict) for msg_dict in msg_list]
                cache_messages[int(k)] = (msg_objs, timestamp)
            self.cache_messages = cache_messages

            # 解析 memory_snapshot
            self.memory_snapshot = [
                MemoryMapping.from_dict(d)
                for d in data.get('memory_snapshot', [])
            ]

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning(f'加载缓存失败：{e}')
            # 出错时回退到默认状态
            self.max_msg_id = -1
            self.cache_messages = {}
            self.memory_snapshot = []

    def _delete_single(self, msg_id: int):
        """删除单个消息及其相关的内存记录

        参数：
            msg_id: 要删除的消息 ID
        """
        messages_to_delete = self.cache_messages.get(msg_id, None)
        if messages_to_delete is None:
            return
        self.cache_messages.pop(msg_id, None)
        if msg_id == self.max_msg_id:
            # 如果删除的是最大 ID，更新 max_msg_id
            self.max_msg_id = max(self.cache_messages.keys())

        idx = 0
        while idx < len(self.memory_snapshot):

            enable_ids = self.memory_snapshot[idx].enable_idxs
            disable_id = self.memory_snapshot[idx].disable_idx
            if msg_id == disable_id:
                # 如果 msg_id 等于禁用 ID，尝试重新启用该内存
                self.memory_snapshot[idx].try_enable(msg_id)
                metadata = {'user_id': self.user_id}
                if self.agent_id:
                    metadata['agent_id'] = self.agent_id
                if self.run_id:
                    metadata['run_id'] = self.run_id
                try:
                    self.memory._create_memory(
                        data=self.memory_snapshot[idx].value,
                        existing_embeddings={},
                        metadata=metadata)
                except Exception as e:
                    logger.warning(f'恢复内存失败：{e}')
            if msg_id in enable_ids:
                if len(enable_ids) > 1:
                    # 如果有多个启用 ID，只移除当前 msg_id
                    self.memory_snapshot[idx].enable_idxs.remove(msg_id)
                else:
                    # 如果只有一个启用 ID，删除内存并移除快照
                    self.memory.delete(self.memory_snapshot[idx].memory_id)
                    self.memory_snapshot.pop(idx)
                    idx -= 1  # pop 后，当前 idx 已经是下一个项目

            idx += 1

    async def add_single(self,
                         messages: List[Message],
                         user_id: Optional[int] = None,
                         agent_id: Optional[int] = None,
                         run_id: Optional[int] = None,
                         memory_type: Optional[str] = None,
                         msg_id: Optional[int] = None) -> None:
        """添加单个消息块到内存缓存

        参数：
            messages: 消息列表
            user_id: 用户 ID
            agent_id: 代理 ID
            run_id: 运行 ID
            memory_type: 内存类型
            msg_id: 消息 ID，如果不提供则自动生成
        """
        messages_dict = []
        for message in messages:
            if isinstance(message, Message):
                messages_dict.append(message.to_dict_clean())
            else:
                messages_dict.append(message)
        async with self._lock:
            if msg_id is None:
                # 自动生成消息 ID
                self.max_msg_id += 1
                msg_id = self.max_msg_id
            else:
                # 使用提供的 ID，并更新 max_msg_id
                self.max_msg_id = max(msg_id, self.max_msg_id)
            self.cache_messages[msg_id] = messages, self._hash_block(messages)

            try:
                self.memory.add(
                    messages_dict,
                    user_id=user_id or self.user_id,
                    agent_id=agent_id or self.agent_id,
                    run_id=run_id or self.run_id,
                    memory_type=memory_type)
                logger.info('添加内存成功。')
            except Exception as e:
                logger.warning(f'添加内存失败：{e}')

            if self.history_mode == 'overwrite':
                # 在覆盖模式下，获取所有内存并进行比对
                res = self.memory.get_all(
                    user_id=user_id or self.user_id,
                    agent_id=agent_id or self.agent_id,
                    run_id=run_id or self.run_id)  # 已排序
                res = [(item['id'], item['memory']) for item in res['results']]
                if len(res):
                    logger.info('所有内存信息：')
                for item in res:
                    logger.info(item[1])
                valids = []
                unmatched = []
                for id, memory in res:
                    matched = False
                    for item in self.memory_snapshot:
                        if id == item.memory_id:
                            if item.value == memory and item.valid:
                                matched = True
                                valids.append(id)
                                break
                            else:
                                if item.valid:
                                    item.disable(msg_id)
                    if not matched:
                        unmatched.append((id, memory))
                for item in self.memory_snapshot:
                    if item.memory_id not in valids:
                        item.disable(msg_id)
                for (id, memory) in unmatched:
                    m = MemoryMapping(
                        memory_id=id, value=memory, enable_idxs=msg_id)
                    self.memory_snapshot.append(m)

    def search(self,
               query: str,
               meta_infos: List[Dict[str, Any]] = None) -> List[str]:
        """
        根据查询字符串和可选的元数据过滤器搜索相关记忆

        此方法使用提供的元数据约束（如 user_id、agent_id、run_id）对内部内存存储执行一次或多次搜索。
        `meta_infos` 中的每个条目定义一个单独的搜索上下文。如果未提供 `meta_infos`，
        则使用实例的默认属性执行默认搜索。

        参数：
            query: 用于语义或基于关键词检索的输入查询字符串
            meta_infos: 指定每个搜索请求的元数据过滤器的列表
                每个字典可能包含：
                    - user_id: 按用户 ID 过滤记忆
                    - agent_id: 按代理 ID 过滤记忆
                    - run_id: 按会话/运行 ID 过滤记忆
                    - limit: 每个搜索返回的最大结果数
                如果为 None，则使用实例级别的值执行单个默认搜索

        返回：
            来自所有搜索结果的扁平化记忆内容字符串列表
            每个字符串代表一个相关的记忆条目

        注意：
            - 对于 meta_info 字典中缺失的字段，使用实例的对应属性作为回退值
        """
        if meta_infos is None:
            meta_infos = [{
                'user_id': self.user_id,
                'agent_id': self.agent_id,
                'run_id': self.run_id,
                'limit': self.search_limit,
            }]
        memories = []
        for meta_info in meta_infos:
            user_id = meta_info.get('user_id', None)
            agent_id = meta_info.get('agent_id', None)
            run_id = meta_info.get('run_id', None)
            limit = meta_info.get('limit', self.search_limit)
            relevant_memories = self.memory.search(
                query,
                user_id=user_id or self.user_id,
                agent_id=agent_id or self.agent_id,
                run_id=run_id or self.run_id,
                limit=limit)
            memories.extend(
                [entry['memory'] for entry in relevant_memories['results']])
        return memories

    def _split_into_blocks(self,
                           messages: List[Message]) -> List[List[Message]]:
        """
        将消息分割成块，每个块以'user'消息开始，
        包含其后所有非'user'消息，直到下一个'user'消息（不包含）

        第一个'user'消息之前的消息（如 system）会附加到第一个用户块中
        如果没有'user'消息，所有消息放入一个块中
        """
        if not messages:
            return []

        blocks: List[List[Message]] = []
        current_block: List[Message] = []

        # 处理前导非用户消息（如 system）
        have_user = False
        for msg in messages:
            if msg.role != 'user':
                current_block.append(msg)
            else:
                if have_user:
                    # 已经遇到过'user'，保存当前块并开始新块
                    blocks.append(current_block)
                    current_block = [msg]
                else:
                    # 第一个'user'消息
                    current_block.append(msg)
                    have_user = True

        # 添加最后一个块
        if current_block:
            blocks.append(current_block)

        return blocks

    def _hash_block(self, block: List[Message]) -> str:
        """计算消息块的 sha256 哈希值用于比较"""
        data = [message.to_dict_clean() for message in block]
        allow_role = ['user', 'system', 'assistant', 'tool']
        allow_role = [
            role for role in allow_role if role not in self.ignore_roles
        ]
        allow_fields = ['reasoning_content', 'content', 'tool_calls', 'role']
        allow_fields = [
            field for field in allow_fields if field not in self.ignore_fields
        ]

        data = [{
            field: value
            for field, value in msg.items() if field in allow_fields
        } for msg in data if msg['role'] in allow_role]

        block_data = json5.dumps(data)
        return hashlib.sha256(block_data.encode('utf-8')).hexdigest()

    def _analyze_messages(
            self,
            messages: List[Message]) -> Tuple[List[List[Message]], List[int]]:
        """
        分析传入消息与缓存的对比关系

        返回：
            should_add_messages: 需要添加的消息块（不在缓存中或哈希值已改变）
            should_delete: 需要删除的消息 ID 列表（在缓存中但不在新块中）
        """
        new_blocks = self._split_into_blocks(messages)
        self.cache_messages = dict(sorted(self.cache_messages.items()))
        cache_messages = [(key, value)
                          for key, value in self.cache_messages.items()]

        first_unmatched_idx = -1

        for idx in range(len(new_blocks)):
            block_hash = self._hash_block(new_blocks[idx])

            # 必须允许比较到最后一个缓存条目
            if idx < len(cache_messages) and str(block_hash) == str(
                    cache_messages[idx][1][1]):
                continue

            # 不匹配
            first_unmatched_idx = idx
            break

        # 如果所有 new_blocks 都匹配，但缓存有多余条目 -> 删除多余的缓存条目
        if first_unmatched_idx == -1:
            should_add_messages = []
            should_delete = [
                item[0] for item in cache_messages[len(new_blocks):]
            ]
            return should_add_messages, should_delete

        # 不匹配时：添加所有新块，并删除从 mismatch 索引开始的所有缓存条目
        should_add_messages = new_blocks[first_unmatched_idx:]
        should_delete = [
            item[0] for item in cache_messages[first_unmatched_idx:]
        ]

        return should_add_messages, should_delete

    def _get_user_message(self, block: List[Message]) -> Optional[Message]:
        """辅助方法：获取块中的用户消息（如果存在）"""
        for msg in block:
            if msg.role == 'user':
                return msg
        return None

    async def add(
        self,
        messages: List[Message],
        user_id: Optional[List[str]] = None,
        agent_id: Optional[List[str]] = None,
        run_id: Optional[List[str]] = None,
        memory_type: Optional[List[str]] = None,
    ) -> None:
        """添加消息到内存

        参数：
            messages: 消息列表
            user_id: 用户 ID 列表
            agent_id: 代理 ID 列表
            run_id: 运行 ID 列表
            memory_type: 内存类型列表
        """
        should_add_messages, should_delete = self._analyze_messages(messages)

        if should_delete:
            if self.history_mode == 'overwrite':
                # 在覆盖模式下，删除需要移除的消息
                for msg_id in should_delete:
                    self._delete_single(msg_id=msg_id)
                res = self.memory.get_all(
                    user_id=user_id or self.user_id,
                    agent_id=agent_id or self.agent_id,
                    run_id=run_id or self.run_id)  # 已排序
                res = [(item['id'], item['memory']) for item in res['results']]
                logger.info('回滚成功。所有内存信息：')
                for item in res:
                    logger.info(item[1])
        if should_add_messages:
            # 添加新的消息块
            for messages in should_add_messages:
                messages = self.parse_messages(messages)
                await self.add_single(
                    messages,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    memory_type=memory_type)
        self.save_cache()

    def parse_messages(self, messages: List[Message]) -> List[Message]:
        """解析消息列表，根据配置过滤掉不需要的角色和字段

        参数：
            messages: 原始消息列表

        返回：
            过滤后的消息列表
        """
        new_messages = []
        for msg in messages:
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)

            if 'system' not in self.ignore_roles and role == 'system':
                new_messages.append(msg)
            if role == 'user':
                new_messages.append(msg)
            if 'assistant' not in self.ignore_roles and role == 'assistant' and content is not None:
                new_messages.append(msg)
            if 'tool' not in self.ignore_roles and role == 'tool':
                new_messages.append(msg)

        return new_messages

    def delete(self,
               user_id: Optional[str] = None,
               agent_id: Optional[str] = None,
               run_id: Optional[str] = None,
               memory_ids: Optional[List[str]] = None) -> Tuple[bool, str]:
        """删除内存记录

        参数：
            user_id: 用户 ID
            agent_id: 代理 ID
            run_id: 运行 ID
            memory_ids: 要删除的内存 ID 列表，如果为 None 则删除所有

        返回：
            (成功标志，错误信息)
        """
        failed = {}
        if memory_ids is None:
            # 删除所有
            try:
                self.memory.delete_all(
                    user_id=user_id, agent_id=agent_id, run_id=run_id)
                return True, ''
            except Exception as e:
                return False, str(e) + '\n' + traceback.format_exc()
        for memory_id in memory_ids:
            try:
                self.memory.delete(memory_id=memory_id)
            except IndexError:
                failed[
                    memory_id] = '该 memory_id 在数据库中不存在。\n' + traceback.format_exc(
                    )
            except Exception as e:
                failed[memory_id] = str(e) + '\n' + traceback.format_exc()
        if failed:
            return False, json.dumps(failed)
        else:
            return True, ''

    def get_all(self,
                user_id: Optional[str] = None,
                agent_id: Optional[str] = None,
                run_id: Optional[str] = None):
        """获取所有内存记录

        参数：
            user_id: 用户 ID
            agent_id: 代理 ID
            run_id: 运行 ID

        返回：
            内存记录列表，出错时返回空列表
        """
        try:
            res = self.memory.get_all(
                user_id=user_id or self.user_id,
                agent_id=agent_id,
                run_id=run_id)
            return res['results']
        except Exception:
            return []

    def _get_latest_user_message(self,
                                 messages: List[Message]) -> Optional[str]:
        """获取最新的用户消息内容"""
        for message in reversed(messages):
            if message.role == 'user' and hasattr(message, 'content'):
                return message.content
        return None

    def _inject_memories_into_messages(self, messages: List[Message],
                                       memories: List[str],
                                       keep_details) -> List[Message]:
        """将相关记忆注入到系统消息中

        参数：
            messages: 原始消息列表
            memories: 搜索到的相关记忆列表
            keep_details: 是否保留详细消息历史

        返回：
            注入记忆后的新消息列表
        """
        # 格式化记忆以便注入
        memories_str = '用户记忆：\n' + '\n'.join(f'- {memory}'
                                                      for memory in memories)
        # 移除与记忆对应的消息部分，并添加相关的 memory_str 信息

        if getattr(messages[0], 'role') == 'system':
            # 如果第一条是系统消息，追加记忆到其内容
            system_prompt = getattr(
                messages[0], 'content') + f'\n用户记忆：{memories_str}'
            remain_idx = 1
        else:
            # 否则创建新的系统消息
            system_prompt = f'\n你是一个有帮助的助手。根据查询和记忆回答问题。\n' \
                            f'\n用户记忆：{memories_str}'
            remain_idx = 0
        if not keep_details:
            # 不保留详细信息时，分析消息以确定保留多少历史
            should_add_messages, should_delete = self._analyze_messages(
                messages)
            remain_idx = max(
                remain_idx,
                len(messages)
                - sum([len(block) for block in should_add_messages]))

        new_messages = [Message(role='system', content=system_prompt)
                        ] + messages[remain_idx:]
        return new_messages

    async def run(
        self,
        messages: List[Message],
        meta_infos: List[Dict[str, Any]] = None,
        keep_details: bool = True,
    ):
        """运行记忆检索并将结果注入到消息中

        参数：
            messages: 输入消息列表
            meta_infos: 搜索元数据信息列表
            keep_details: 是否保留详细消息历史

        返回：
            可能包含注入记忆的消息列表
        """
        if not self.is_retrieve:
            # 如果未启用检索，直接返回原消息
            return messages

        query = self._get_latest_user_message(messages)
        if not query:
            # 没有用户查询，返回原消息
            return messages
        async with self._lock:
            try:
                memories = self.search(query, meta_infos)
            except Exception as search_error:
                logger.warning(f'搜索记忆失败：{search_error}')
                memories = []
            if memories:
                # 有相关记忆，注入到消息中
                messages = self._inject_memories_into_messages(
                    messages, memories, keep_details)
            return messages

    def _init_memory_obj(self):
        """初始化内存对象（使用 mem0 库）"""
        try:
            import mem0
        except ImportError as e:
            logger.error(
                f'导入 mem0 失败：{e}。请通过 `pip install mem0ai` 安装 mem0 包。'
            )
            raise

        capture_event_origin = mem0.memory.main.capture_event

        @wraps(capture_event_origin)
        def patched_capture_event(event_name,
                                  memory_instance,
                                  additional_data=None):
            # 禁用事件捕获
            pass

        mem0.memory.main.capture_event = partial(patched_capture_event, )

        # embedder 配置
        embedder = None
        embedder_config = getattr(self.config.memory.default_memory,
                                  'embedder', OmegaConf.create({}))
        service = getattr(embedder_config, 'service', 'modelscope')
        api_key = getattr(embedder_config, 'api_key', None)
        emb_model = getattr(embedder_config, 'model',
                            'Qwen/Qwen3-Embedding-8B')
        embedding_dims = getattr(embedder_config, 'embedding_dims',
                                 None)  # 用于向量存储配置

        if self.is_retrieve:
            # 启用检索时配置 embedder
            embedder = OmegaConf.create({
                'provider': 'openai',
                'config': {
                    'api_key': api_key
                    or os.getenv(f'{service.upper()}_API_KEY'),
                    'openai_base_url': get_service_config(service).base_url,
                    'model': emb_model,
                    'embedding_dims': embedding_dims
                }
            })

        # llm 配置（用于压缩）
        llm = None
        if self.compress:
            llm_config = getattr(self.config, 'llm', None)
            if llm_config is not None:
                service = getattr(llm_config, 'service', 'modelscope')
                llm_model = getattr(llm_config, 'model',
                                    'Qwen/Qwen3-Coder-30B-A3B-Instruct')
                api_key = getattr(llm_config, f'{service}_api_key', None)
                openai_base_url = getattr(llm_config, f'{service}_base_url',
                                          None)
                gen_cfg = getattr(self.config, 'generation_config', None)
                max_tokens = getattr(gen_cfg, 'max_tokens', None)

                llm = {
                    'provider': 'openai',
                    'config': {
                        'model':
                        llm_model,
                        'api_key':
                        api_key or os.getenv(f'{service.upper()}_API_KEY'),
                        'openai_base_url':
                        openai_base_url
                        or get_service_config(service).base_url,
                    }
                }
                if max_tokens is not None:
                    llm['config']['max_tokens'] = max_tokens

        # 向量存储配置
        def sanitize_database_name(ori_name: str,
                                   default_name: str = 'default') -> str:
            """清理数据库名称，使其符合命名规范"""
            if not ori_name or not isinstance(ori_name, str):
                return default_name
            sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', ori_name)
            sanitized = re.sub(r'_+', '_', sanitized)
            sanitized = sanitized.strip('_')
            if not sanitized:
                return default_name
            if sanitized[0].isdigit():
                sanitized = f'col_{sanitized}'
            return sanitized

        vector_store_config = getattr(self.config.memory.default_memory,
                                      'vector_store', OmegaConf.create({}))
        vector_store_provider = getattr(vector_store_config, 'service',
                                        'qdrant')
        on_disk = getattr(vector_store_config, 'on_disk', True)
        path = getattr(vector_store_config, 'path', self.path)
        db_name = getattr(vector_store_config, 'db_name', None)
        url = getattr(vector_store_config, 'url', None)
        token = getattr(vector_store_config, 'token', None)
        collection_name = getattr(vector_store_config, 'collection_name', path)

        db_name = sanitize_database_name(db_name) if db_name else None
        collection_name = sanitize_database_name(
            collection_name) if collection_name else None

        # 检查值
        from mem0.memory.main import VectorStoreFactory
        class_type = VectorStoreFactory.provider_to_class.get(
            vector_store_provider)
        if class_type:
            module_path, class_name = class_type.rsplit('.', 1)
            module = importlib.import_module(module_path)
            vector_store_class = getattr(module, class_name)
            parameters = signature(vector_store_class.__init__).parameters

            config_raw = {
                'path': path,
                'on_disk': on_disk,
                'collection_name': collection_name,
                'url': url,
                'token': token,
                'db_name': db_name,
                'embedding_model_dims': embedding_dims
            }
            config_format = {
                key: value
                for key, value in config_raw.items()
                if value and key in parameters
            }
            vector_store = {
                'provider': vector_store_provider,
                'config': config_format
            }
        else:
            vector_store = {}

        mem0_config = {'is_infer': self.compress, 'vector_store': vector_store}
        if embedder:
            mem0_config['embedder'] = embedder
        if llm:
            mem0_config['llm'] = llm
        logger.info(f'内存配置：{mem0_config}')
        # Prompt 内容过长，默认日志记录会降低可读性
        custom_fact_extraction_prompt = getattr(
            self.config.memory.default_memory, 'fact_retrieval_prompt',
            getattr(self.config.memory.default_memory,
                    'custom_fact_extraction_prompt', None))
        if custom_fact_extraction_prompt is not None:
            mem0_config['custom_fact_extraction_prompt'] = (
                custom_fact_extraction_prompt
                + f'今天的日期是 {datetime.now().strftime("%Y-%m-%d")}。')
        try:
            memory = mem0.Memory.from_config(mem0_config)
            memory._telemetry_vector_store = None
        except Exception as e:
            logger.error(f'初始化 Mem0 内存失败：{e}')
            # 此处不抛出异常，仅记录日志并继续，内存在不可用时将为 None
            memory = None
        return memory
