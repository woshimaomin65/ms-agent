# Disable frozen modules warning for debugger
# This must be set before any other imports
import glob
import os
import shutil
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Clean up old temp directories from previous runs
for old_dir in glob.glob('/tmp/tmp*/agent'):
    try:
        shutil.rmtree(os.path.dirname(old_dir))
        print(f'Cleaned up old temp directory: {os.path.dirname(old_dir)}')
    except Exception:
        pass

import ast
import asyncio
import re
import tempfile
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Pattern, Optional, Tuple

from ms_agent.config.config import Config
from ms_agent.workflow.loader import WorkflowLoader
from ms_agent.workflow.dag_workflow import DagWorkflow
from omegaconf import DictConfig, OmegaConf

# ============== 配置部分 ==============
TARGET_DIR = '/Users/maomin/programs/vscode/ms-agent/ms_agent'
BATCH_SIZE = 5  # 最大并发子任务数

def find_python_files(root_dir: str) -> List[str]:
    """递归查找所有 Python 文件"""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过常见排除目录
        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'venv', '.venv', 'node_modules', 'dist'}]
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return sorted(py_files)

# Unicode range for CJK (Chinese-Japanese-Korean) characters
CJK_UNICODE_PATTERN: Pattern[str] = re.compile(r'[\u4e00-\u9fff]')


@dataclass
class CommentCoverage:
    """Represents the Chinese comment coverage analysis result for a file."""
    file_path: str
    has_chinese_comments: bool
    total_items: int  # Total number of functions/classes/methods
    commented_items: int  # Number of items with Chinese docstrings
    has_module_docstring: bool  # Whether module has Chinese docstring
    coverage_ratio: float  # Ratio of commented items (0.0 to 1.0)
    error: Optional[str] = None  # Error message if parsing failed
    
    @property
    def is_fully_commented(self) -> bool:
        """Check if the file is fully commented with Chinese."""
        if self.error:
            return False
        if self.total_items == 0:
            # No functions/classes, check module docstring only
            return self.has_module_docstring
        # Consider fully commented if all items have Chinese docstrings
        # and module has Chinese docstring (if it exists)
        return self.commented_items == self.total_items


def _has_chinese_text(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(CJK_UNICODE_PATTERN.search(text)) if text else False


def _get_docstring(node: ast.AST) -> Optional[str]:
    """
    Extract docstring from an AST node (function, class, or module).
    
    Handles both Python 3.8+ (docstring in body[0]) and older styles.
    """
    if isinstance(node, ast.Module):
        # Module level docstring
        if node.body and isinstance(node.body[0], ast.Expr):
            expr = node.body[0]
            if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                return expr.value.value
            # Python 3.7 compatibility
            elif isinstance(expr.value, ast.Str):
                return expr.value.s
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # Function/class level docstring
        if node.body and isinstance(node.body[0], ast.Expr):
            expr = node.body[0]
            if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                return expr.value.value
            # Python 3.7 compatibility
            elif isinstance(expr.value, ast.Str):
                return expr.value.s
    return None


def _count_code_items(tree: ast.AST) -> Tuple[int, int]:
    """
    Count total code items (functions, classes, methods) and those with Chinese docstrings.
    
    Returns:
        Tuple of (total_items, commented_items)
    """
    total = 0
    commented = 0
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total += 1
            docstring = _get_docstring(node)
            if docstring and _has_chinese_text(docstring):
                commented += 1
    
    return total, commented


def analyze_chinese_comments(file_path: str) -> CommentCoverage:
    """
    Analyze Chinese comment coverage in a Python file using AST parsing.
    
    This function parses the Python file and checks:
    1. Whether the module has a Chinese docstring
    2. How many functions/classes have Chinese docstrings
    3. The overall coverage ratio
    
    This approach correctly handles partially commented files. For example,
    if a 1000-line file has Chinese docstrings for only the first 500 lines
    (some functions), it will return the exact coverage ratio.
    
    Args:
        file_path: Path to the Python file to analyze.
        
    Returns:
        CommentCoverage object with detailed analysis results.
        
    Example:
        >>> result = analyze_chinese_comments('example.py')
        >>> print(f"Coverage: {result.coverage_ratio:.1%}")
        >>> if not result.is_fully_commented:
        ...     print(f"Need to comment {result.total_items - result.commented_items} items")
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return CommentCoverage(
                file_path=file_path,
                has_chinese_comments=False,
                total_items=0,
                commented_items=0,
                has_module_docstring=False,
                coverage_ratio=0.0
            )
        
        # Parse the file into AST
        tree = ast.parse(content, filename=file_path)
        
        # Check module docstring
        module_docstring = _get_docstring(tree)
        has_module_docstring = bool(module_docstring and _has_chinese_text(module_docstring))
        
        # Count functions/classes and their docstrings
        total_items, commented_items = _count_code_items(tree)
        
        # Calculate coverage ratio
        if total_items == 0:
            coverage_ratio = 1.0 if has_module_docstring else 0.0
        else:
            coverage_ratio = commented_items / total_items
        
        # Check if there's any Chinese comment (including line comments)
        has_any_chinese = has_module_docstring or commented_items > 0
        if not has_any_chinese:
            # Also check for Chinese in line comments as fallback
            has_any_chinese = bool(CJK_UNICODE_PATTERN.search(content))
        
        return CommentCoverage(
            file_path=file_path,
            has_chinese_comments=has_any_chinese,
            total_items=total_items,
            commented_items=commented_items,
            has_module_docstring=has_module_docstring,
            coverage_ratio=coverage_ratio
        )
        
    except SyntaxError as e:
        return CommentCoverage(
            file_path=file_path,
            has_chinese_comments=False,
            total_items=0,
            commented_items=0,
            has_module_docstring=False,
            coverage_ratio=0.0,
            error=f"Syntax error: {e}"
        )
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, OSError) as e:
        return CommentCoverage(
            file_path=file_path,
            has_chinese_comments=False,
            total_items=0,
            commented_items=0,
            has_module_docstring=False,
            coverage_ratio=0.0,
            error=f"Read error: {e}"
        )


def has_chinese_comments(file_path: str) -> bool:
    """
    Check if a Python file needs Chinese comments added.
    
    This function uses AST parsing to determine if a file is already fully
    commented with Chinese. A file is considered "commented" only if:
    1. All functions/classes have Chinese docstrings
    2. The module has a Chinese docstring (if applicable)
    
    Files with partial comments (e.g., only some functions commented) will
    return False, indicating they need more comments.
    
    Args:
        file_path: Path to the Python file to check.
        
    Returns:
        True if the file is fully commented with Chinese, False otherwise.
        Also returns False if the file cannot be read or parsed.
        
    Note:
        This function is designed for the annotation workflow to skip files
        that are already fully commented. Files with partial comments will
        still be processed to add missing docstrings.
    """
    coverage = analyze_chinese_comments(file_path)
    return coverage.is_fully_commented

def create_base_agent_config():
    """创建基础 Agent 配置"""
    return {
        'llm': {
            'service': 'dashscope',
            'model': 'qwen-plus',
            'dashscope_api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6cacbd1fc53f4c8ebd80fdfcfe75a533'),
            'dashscope_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'modelscope_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        },
        'generation_config': {
            'temperature': 0.3,
            'top_p': 0.9,
            'top_k': 50,
            'max_tokens': 8192,
            'stream': False,
            'show_reasoning': False,
        },
        'max_chat_round': 10,
        'save_history': False,
        'tools': {
            'file_system': {
                'enabled': True,
                'mcp': False,
                'allow_read_all_files': True,
            },
            'shell': {
                'enabled': True,
                'mcp': False,
            },
            'code_executor': {
                'implementation': 'simple',
                'mcp': False,
            },
            'todo_list': {
                'enabled': True,
                'mcp': False,
            },
        },
    }

def create_workflow_config(py_files: List[str], batch_size: int = 5):
    """
    创建 DagWorkflow 配置
    
    工作流结构:
    1. scanner: 扫描文件列表（已完成，传入）
    2. dispatcher: 分发任务到 subagent_N
    3. subagent_1 ~ subagent_N: 并行处理文件注释
    """
    # 过滤掉已有中文注释的文件
    files_to_annotate = [f for f in py_files if not has_chinese_comments(f)]
    print(f"找到 {len(py_files)} 个 Python 文件")
    print(f"需要添加注释的文件：{len(files_to_annotate)} 个")
    
    if not files_to_annotate:
        print("所有文件都已有中文注释！")
        return None, []
    
    # 分批处理
    batches = []
    for i in range(0, len(files_to_annotate), batch_size):
        batches.append(files_to_annotate[i:i + batch_size])
    
    print(f"分为 {len(batches)} 批处理，每批最多 {batch_size} 个文件")
    
    # 创建工作流配置
    workflow_config = {
        'type': 'DagWorkflow',
    }
    
    # Scanner 任务 - 实际上我们已经预先扫描了
    workflow_config['scanner'] = {
        'next': ['dispatcher'],
        'agent_config': 'agent/scanner_agent.yaml',
        'agent': {
            'name': 'LLMAgent',
            'kwargs': {'tag': 'scanner'}
        }
    }
    
    # Dispatcher 任务
    workflow_config['dispatcher'] = {
        'next': [],  # 会动态添加 subagent
        'agent_config': 'agent/dispatcher_agent.yaml',
        'agent': {
            'name': 'LLMAgent',
            'kwargs': {'tag': 'dispatcher'}
        }
    }
    
    # 为每个批次创建 subagent
    for batch_idx, batch_files in enumerate(batches):
        subagent_name = f'subagent_{batch_idx}'
        workflow_config[subagent_name] = {
            'agent_config': 'agent/subagent.yaml',
            'agent': {
                'name': 'LLMAgent',
                'kwargs': {
                    'tag': subagent_name,
                }
            }
        }
        # 添加到 dispatcher 的 next
        workflow_config['dispatcher']['next'].append(subagent_name)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 保存工作流配置
    workflow_file = os.path.join(temp_dir, 'workflow.yaml')
    with open(workflow_file, 'w', encoding='utf-8') as f:
        yaml.dump(workflow_config, f, default_flow_style=False, allow_unicode=True)
    
    # 创建 Agent 配置
    agent_dir = os.path.join(temp_dir, 'agent')
    os.makedirs(agent_dir, exist_ok=True)
    
    # Scanner Agent 配置
    scanner_config = create_base_agent_config()
    scanner_config['prompt'] = {
        'system': f'''你是一个文件扫描助手。
你的任务是扫描目录 {TARGET_DIR} 下的所有 Python 文件。
已扫描到的文件列表：
{chr(10).join(py_files)}

请输出这个文件列表供后续处理。''',
        'query': '请扫描并输出文件列表'
    }
    with open(os.path.join(agent_dir, 'scanner_agent.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(scanner_config, f, default_flow_style=False, allow_unicode=True)
    
    # Dispatcher Agent 配置
    dispatcher_config = create_base_agent_config()
    dispatcher_config['prompt'] = {
        'system': f'''你是一个任务分发助手。
你需要将文件注释任务分发给多个 subagent 并行处理。
每批处理的文件数量不超过 {batch_size} 个。

文件批次：
{chr(10).join([f"批次 {i}: {batch}" for i, batch in enumerate(batches)])}

请将每个批次的任务分发给对应的 subagent。''',
        'query': '请分发任务给各个 subagent'
    }
    with open(os.path.join(agent_dir, 'dispatcher_agent.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(dispatcher_config, f, default_flow_style=False, allow_unicode=True)
    
    # SubAgent 配置 - 通用的文件注释 Agent
    subagent_config = create_base_agent_config()
    subagent_config['prompt'] = {
        'system': '''你是一个专业的代码注释助手。

你的任务：
1. 读取给定的 Python 文件
2. 分析代码结构和功能
3. 为代码添加清晰的中文注释
4. 保持原有代码风格和功能不变

注释要求：
- 模块级别：添加文件功能说明
- 类级别：添加类的作用说明
- 函数/方法级别：添加参数说明、返回值说明、功能描述
- 复杂逻辑：添加行内注释解释关键步骤

注意事项：
- 不要改变原有代码逻辑
- 保持注释简洁明了
- 使用专业的技术术语
- 如果文件已有部分中文注释，补充缺失的部分

请用中文回复，并直接输出修改后的完整代码。''',
        'query': '请为给定的 Python 文件添加中文注释'
    }
    with open(os.path.join(agent_dir, 'subagent.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(subagent_config, f, default_flow_style=False, allow_unicode=True)
    
    return temp_dir, files_to_annotate

class AnnotateDagWorkflow(DagWorkflow):
    """自定义 DagWorkflow，支持文件注释任务"""
    
    WORKFLOW_NAME = 'AnnotateDagWorkflow'
    
    def __init__(self, py_files: List[str], batch_size: int = BATCH_SIZE, **kwargs):
        # 先创建工作流配置
        config_dir, files_to_annotate = create_workflow_config(py_files, batch_size)
        
        if config_dir is None:
            self.config = None
            self.files_to_annotate = []
            return
            
        self.files_to_annotate = files_to_annotate
        self.config_dir = config_dir
        
        # 加载配置
        config = Config.from_task(config_dir)
        
        super().__init__(config=config, **kwargs)
        
        # 存储文件批次信息
        self.batches = []
        for i in range(0, len(files_to_annotate), batch_size):
            self.batches.append(files_to_annotate[i:i + batch_size])
    
    async def run(self, inputs: Any = None, **kwargs):
        """运行工作流"""
        if self.config is None:
            print("无需处理，所有文件都已有中文注释")
            return {}
        
        print(f"\n开始处理 {len(self.files_to_annotate)} 个文件...")
        print(f"批次数量：{len(self.batches)}")
        print(f"文件列表:")
        for i, f in enumerate(self.files_to_annotate[:10]):  # 只显示前 10 个
            print(f"  {i+1}. {f}")
        if len(self.files_to_annotate) > 10:
            print(f"  ... 还有 {len(self.files_to_annotate) - 10} 个文件")
        
        # 调用父类运行
        return await super().run(inputs, **kwargs)

async def main():
    print("=" * 60)
    print("Python 文件中文注释工作流")
    print("=" * 60)
    
    # 1. 扫描所有 Python 文件
    print(f"\n【步骤 1】扫描 {TARGET_DIR} 目录...")
    py_files = find_python_files(TARGET_DIR)
    print(f"找到 {len(py_files)} 个 Python 文件")
    
    # 2. 检查工作注释状态
    print("\n【步骤 2】检查文件注释状态...")
    commented_count = 0
    not_commented_count = 0
    for f in py_files[:20]:  # 只检查前 20 个文件
        has_comments = has_chinese_comments(f)
        status = "✅ 已注释" if has_comments else "❌ 未注释"
        if has_comments:
            commented_count += 1
        else:
            not_commented_count += 1
        print(f"  {status}: {f}")
    if len(py_files) > 20:
        print(f"  ... 还有 {len(py_files) - 20} 个文件未显示")
    
    # 3. 创建工作流
    print("\n【步骤 3】创建工作流配置...")
    workflow = AnnotateDagWorkflow(
        py_files=py_files,
        batch_size=BATCH_SIZE,
        trust_remote_code=True,
        load_cache=False
    )
    
    if workflow.config is None:
        print("\n✅ 所有文件都已有中文注释，无需处理！")
        return
    
    print(f"\n工作流配置目录：{workflow.config_dir}")
    print(f"需要处理的文件数：{len(workflow.files_to_annotate)}")
    
    # 4. 运行工作流
    print("\n【步骤 4】运行工作流...")
    print("=" * 60)
    
    try:
        result = await workflow.run(None)
        
        print("\n" + "=" * 60)
        print("✅ 工作流执行完成！")
        print(f"结果：{result}")
        
    except Exception as e:
        print(f"\n❌ 工作流执行出错：{e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    asyncio.run(main())
