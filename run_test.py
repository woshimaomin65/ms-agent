import asyncio
import os

from ms_agent import LLMAgent
from omegaconf import DictConfig

# Configure MCP servers
# 注意：确保这个 MCP 服务器 URL 是有效的
# 你可以从 ModelScope MCP Playground 获取有效的服务器 URL
def create_config():
    """创建配置对象

    配置说明:
    - 使用阿里云 DashScope 服务 (百炼平台)
    - 模型：qwen-plus
    - 支持流式输出和推理内容展示
    """
    config = DictConfig({
        'llm': {
            'service': 'dashscope',
            'model': 'qwen-plus',
            # API Key 优先级：
            # 1. 这里配置的值
            # 2. 环境变量 DASHSCOPE_API_KEY
            # 3. 环境变量 MODELSCOPE_API_KEY
            'dashscope_api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6cacbd1fc53f4c8ebd80fdfcfe75a533'),
            # DashScope 兼容模式 v1 端点
            'dashscope_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            # 添加 modelscope_base_url 以避免 DashScope 类初始化时出错
            'modelscope_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        },
        'generation_config': {
            'temperature': 0.7,      # 创造性参数，越高越有创意
            'top_p': 0.9,            # 核采样参数
            'top_k': 50,             # 前 K 个候选 token
            'max_tokens': 2048,      # 最大输出长度
            'stream': False,         # 是否启用流式输出
            'show_reasoning': False, # 是否展示推理过程 (thinking)
        },
        'prompt': {
            'system': '''你是一个有帮助的 AI 助手。请遵循以下原则：

1. 用与用户相同的语言回复
2. 提供准确、详细的答案
3. 如果有不确定的地方，请说明
4. 对于代码相关问题，提供完整的代码示例和解释

请开始帮助用户。'''
        },
        'max_chat_round': 10,        # 最大对话轮数
        'save_history': False,       # 是否保存对话历史

        # 工具配置
        'tools': {
            # 启用文件系统工具（本地工具，不是 MCP 服务器）
            'file_system': {
                'enabled': True,
                'mcp': False,  # 重要：标记为本地工具
                'allow_read_all_files': True,  # 允许读取所有文件（根据需要调整）
            },
            # 启用 Shell 命令工具（本地工具，不是 MCP 服务器）
            'shell': {
                'enabled': True,
                'mcp': False,  # 重要：标记为本地工具
            },
            # 启用代码执行工具 (Python 本地环境)
            'code_executor': {
                'implementation': 'python_env',  # 或 'sandbox'
                'mcp': False,  # 重要：标记为本地工具
            },
            # 启用 Todo 列表工具（本地工具，不是 MCP 服务器）
            'todo_list': {
                'enabled': True,
                'mcp': False,  # 重要：标记为本地工具
            },
        },
    })
    return config

async def main(config):
    #llm_agent = LLMAgent(config=config, mcp_config=mcp)
    llm_agent = LLMAgent(config=config,  trust_remote_code=True)
    
    try:
        #await llm_agent.run('Introduce modelscope.cn用中文介绍')
        await llm_agent.run('/Users/maomin/programs/vscode/ms-agent/ms_agent  检查这个文件夹下的所有文件，包括子文件夹，对其中所有的python 代码添加中文注释， 已经添加过的可以跳过，注意添加注释前请先对整个项目做一定的了解， 清楚代码间的调用关系，先理出执行树或依赖关系后 再开始添加中文注释')
    except Exception as e:
        print(f"Error during agent run: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(llm_agent, 'tool_manager') and llm_agent.tool_manager:
            await llm_agent.tool_manager.cleanup()

if __name__ == '__main__':
    # Start
    config = create_config()
    asyncio.run(main(config))
