# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from ms_agent.agent.loader import AgentLoader
from ms_agent.utils import get_logger
from ms_agent.workflow.base import Workflow
from omegaconf import DictConfig

logger = get_logger()


class ChainWorkflow(Workflow):
    """A workflow implementation that executes tasks in a sequential chain."""

    WORKFLOW_NAME = 'ChainWorkflow'

    def build_workflow(self):
        """根据配置构建任务执行链，解析出有序的任务列表存入 self.workflow_chains。

        算法逻辑：
        1. 遍历所有任务，收集"被某个任务指向的后继任务"集合 has_next。
        2. 不在 has_next 中的任务即为链的起点（start_task）。
        3. 从起点开始，沿 next 指针依次追加，构建有序任务列表。
        """
        if not self.config:
            return

        # 收集所有"被指向"的任务名，用于找到没有前驱的起点
        has_next = set()
        start_task = None
        for task_name, task_config in self.config.items():
            if 'next' in task_config:
                next_tasks = task_config['next']
                if isinstance(next_tasks, str):
                    # next 是单个字符串
                    has_next.add(next_tasks)
                else:
                    # next 是列表，但 ChainWorkflow 只允许一个后继
                    assert len(
                        next_tasks
                    ) == 1, 'ChainWorkflow only supports one next task'
                    has_next.update(next_tasks)

        # 没有被任何任务指向的节点即为链的起点
        for task_name in self.config.keys():
            if task_name not in has_next:
                start_task = task_name
                break

        if start_task is None:
            raise ValueError('No start task found')

        # 从起点出发，沿 next 指针依次追加，构建有序执行链
        result = []
        current_task = start_task

        while current_task:
            result.append(current_task)
            next_task = None
            task_config = self.config[current_task]
            if 'next' in task_config:
                next_tasks = task_config['next']
                if isinstance(next_tasks, str):
                    next_task = next_tasks
                else:
                    next_task = next_tasks[0]

            current_task = next_task

        # 最终有序任务列表，run() 方法将按此顺序执行
        self.workflow_chains = result

    async def run(self, inputs, **kwargs):
        """按顺序执行任务链中的每个任务。

        执行流程：
        - 按 workflow_chains 顺序逐个实例化并运行 Agent。
        - 每一步的输出作为下一步的输入（流水线模式）。
        - 支持通过 next_flow() 实现回跳（循环）逻辑。

        Args:
            inputs (Any): 第一个任务的初始输入。
            **kwargs: 额外参数，透传给各 Agent 的 run 方法。

        Returns:
            Any: 最后一个任务执行完毕后的输出结果。
        """
        agent_config = None
        idx = 0  # 当前执行的任务在 workflow_chains 中的索引

        # 保存每一步开始时的 (inputs, config)，供回跳时恢复现场使用
        step_inputs = {}

        while True:
            task = self.workflow_chains[idx]
            task_info = getattr(self.config, task)

            # 优先使用当前任务自身的 agent_config，否则沿用上一步的配置
            config = getattr(task_info, 'agent_config', agent_config)

            # 确保 agent 字段存在，避免后续 getattr 报错
            if not hasattr(task_info, 'agent'):
                task_info.agent = DictConfig({})

            # 构造 Agent 初始化参数
            init_args = getattr(task_info.agent, 'kwargs', {})
            init_args.pop('trust_remote_code', None)
            init_args['trust_remote_code'] = self.trust_remote_code
            init_args['mcp_server_file'] = self.mcp_server_file
            init_args['task'] = task
            init_args['load_cache'] = self.load_cache
            if isinstance(config, str):
                # config 是路径字符串，拼接本地目录
                init_args['config_dir_or_id'] = os.path.join(
                    self.config.local_dir, config)
            else:
                # config 是 DictConfig 对象，直接传入
                init_args['config'] = config
            init_args['env'] = self.env
            if 'tag' not in init_args:
                init_args['tag'] = task

            # 根据参数动态构建 Agent 实例
            engine = AgentLoader.build(**init_args)

            # 记录当前步骤的输入快照，供后续回跳恢复
            step_inputs[idx] = (inputs, config)

            logger.info(f'Executing step: {task}')
            outputs = await engine.run(inputs)

            # next_flow() 默认返回 idx+1（顺序执行）；
            # 子类可重写该方法返回更小的索引，实现循环回跳。
            # assert 保证每次最多前进一步（不允许跳过任务）。
            next_idx = engine.next_flow(idx)
            assert next_idx - idx <= 1

            if next_idx == idx + 1:
                # 正常前进：将本步输出传给下一步
                inputs = outputs
                agent_config = engine.config
            else:
                # 回跳：恢复目标步骤开始时的输入和配置，重新执行
                inputs, agent_config = step_inputs[next_idx]

            idx = next_idx

            # ---- 退出条件 ----
            # idx 超出任务链长度时退出，即所有预定义任务均已执行完毕。
            # 注意：这里并不检查任务是否"成功完成"，仅判断索引是否越界。
            # 如果某步 Agent 执行失败但未抛出异常，循环仍会继续到下一步。
            if idx >= len(self.workflow_chains):
                break

        return inputs
