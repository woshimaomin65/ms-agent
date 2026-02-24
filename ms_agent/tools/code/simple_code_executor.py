# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Simple code execution tool using subprocess.

This tool executes Python code by:
1. Writing code to a temporary file
2. Running it with subprocess
3. Capturing stdout/stderr
4. Deleting the temp file

No Docker, no ipykernel - just simple subprocess execution.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleCodeExecutionTool(ToolBase):
    """
    Simple code execution tool using subprocess.
    
    Features:
    - Executes Python code in a separate process
    - Captures stdout and stderr
    - Supports timeout
    - Cleans up temp files automatically
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._timeout = getattr(config, 'code_timeout', 60)  # Default 60s timeout
        self._temp_dir = getattr(config, 'code_temp_dir', None)  # Custom temp dir
        logger.info('SimpleCodeExecutionTool initialized (subprocess based)')
    
    async def connect(self) -> None:
        """No connection needed for subprocess execution."""
        logger.info('[SimpleCodeExecutionTool] Connected (no setup required)')
    
    async def cleanup(self) -> None:
        """Clean up any remaining temp files."""
        logger.info('[SimpleCodeExecutionTool] Cleanup completed')
    
    async def get_tools(self) -> Dict[str, Any]:
        """Return tool definition for LLM."""
        return {
            'simple_code_executor': [
                Tool(
                    tool_name='execute_python',
                    server_name='simple_code_executor',
                    description='Execute Python code using subprocess. The code will be written to a temp file and executed.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'code': {
                                'type': 'string',
                                'description': 'Python code to execute'
                            },
                            'timeout': {
                                'type': 'integer',
                                'description': 'Execution timeout in seconds (default: 60)'
                            }
                        },
                        'required': ['code']
                    }
                )
            ]
        }
    
    async def call_tool(self, server_name: str, *, tool_name: str,
                       arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the requested tool."""
        if tool_name == 'execute_python':
            code = arguments.get('code', '')
            timeout = arguments.get('timeout', self._timeout)
            return await self.execute_python(code, timeout)
        else:
            return {
                'success': False,
                'error': f'Unknown tool: {tool_name}'
            }
    
    async def execute_python(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Execute Python code using subprocess.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dict with 'success', 'output', 'error' keys
        """
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(
            suffix='.py',
            prefix='code_exec_',
            dir=self._temp_dir
        )
        os.close(fd)  # Close the file descriptor
        
        try:
            # Write code to temp file
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f'[SimpleCodeExecutionTool] Executing code in {temp_path}')
            
            # Run the code using subprocess
            # Use the same Python interpreter as the current process
            python_executable = sys.executable
            
            try:
                result = subprocess.run(
                    [python_executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.path.dirname(temp_path) or '.'
                )
                
                output = result.stdout
                error = result.stderr
                success = result.returncode == 0
                
                if not success:
                    logger.warning(f'[SimpleCodeExecutionTool] Code execution failed with return code {result.returncode}')
                    if error:
                        logger.error(f'[SimpleCodeExecutionTool] stderr: {error}')
                
                return {
                    'success': success,
                    'output': output,
                    'error': error,
                    'returncode': result.returncode
                }
                
            except subprocess.TimeoutExpired:
                logger.error(f'[SimpleCodeExecutionTool] Code execution timed out after {timeout}s')
                return {
                    'success': False,
                    'output': '',
                    'error': f'Execution timed out after {timeout} seconds'
                }
            except Exception as e:
                logger.error(f'[SimpleCodeExecutionTool] Execution error: {e}')
                return {
                    'success': False,
                    'output': '',
                    'error': f'Execution error: {str(e)}\n{traceback.format_exc()}'
                }
                
        except Exception as e:
            logger.error(f'[SimpleCodeExecutionTool] Failed to create temp file: {e}')
            return {
                'success': False,
                'output': '',
                'error': f'Failed to create temp file: {str(e)}\n{traceback.format_exc()}'
            }
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug(f'[SimpleCodeExecutionTool] Cleaned up temp file: {temp_path}')
            except Exception as e:
                logger.warning(f'[SimpleCodeExecutionTool] Failed to clean up temp file: {e}')
    
    # Convenience methods for direct calling
    async def execute(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Direct method to execute code."""
        return await self.execute_python(code, timeout)
