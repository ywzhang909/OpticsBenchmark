"""
OptiS Benchmark - Base Environment Module

This module defines the base environment interface for optical design tasks.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EnvironmentConfig:
    """Configuration for an environment."""
    name: str
    timeout: int = 300
    max_steps: int = 50
    memory_limit: str = "4GB"
    workspace: Path = Path("/tmp/optis_workspace")
    
    def __post_init__(self):
        """Ensure workspace is a Path."""
        if isinstance(self.workspace, str):
            self.workspace = Path(self.workspace)


@dataclass
class EnvironmentResponse:
    """Response from an environment action."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class BaseEnvironment(ABC):
    """
    Base class for all environments.
    
    Environments provide the execution context for agents,
    including file system access, process execution, and
    integration with specialized software.
    """
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize the environment with configuration."""
        self.config = config
        self.workspace = config.workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._setup_complete = False
    
    @abstractmethod
    async def setup(self) -> None:
        """Set up the environment (e.g., start services, create workspace)."""
        pass
    
    @abstractmethod
    async def teardown(self) -> None:
        """Clean up the environment (e.g., stop services, cleanup files)."""
        pass
    
    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> EnvironmentResponse:
        """
        Execute a command in the environment.
        
        Args:
            command: Command to execute
            timeout: Optional timeout override
            
        Returns:
            EnvironmentResponse with execution results
        """
        pass
    
    async def read_file(self, path: str | Path) -> EnvironmentResponse:
        """
        Read a file from the environment.
        
        Args:
            path: Path to the file
            
        Returns:
            EnvironmentResponse with file contents
        """
        import time
        start = time.time()
        
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace / file_path
            
            if not file_path.exists():
                return EnvironmentResponse(
                    success=False,
                    error=f"File not found: {file_path}",
                    execution_time=time.time() - start,
                )
            
            content = file_path.read_text(encoding="utf-8")
            return EnvironmentResponse(
                success=True,
                output=content,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    async def write_file(
        self,
        path: str | Path,
        content: str,
    ) -> EnvironmentResponse:
        """
        Write content to a file in the environment.
        
        Args:
            path: Path to the file
            content: Content to write
            
        Returns:
            EnvironmentResponse indicating success/failure
        """
        import time
        start = time.time()
        
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace / file_path
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            
            return EnvironmentResponse(
                success=True,
                output=str(file_path),
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    async def list_files(
        self,
        path: str | Path = ".",
        pattern: str = "*",
    ) -> EnvironmentResponse:
        """
        List files in a directory.
        
        Args:
            path: Directory path
            pattern: Glob pattern
            
        Returns:
            EnvironmentResponse with list of files
        """
        import time
        start = time.time()
        
        try:
            dir_path = Path(path)
            if not dir_path.is_absolute():
                dir_path = self.workspace / dir_path
            
            files = [str(p) for p in dir_path.glob(pattern)]
            
            return EnvironmentResponse(
                success=True,
                output=files,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    def get_available_actions(self) -> list[dict[str, Any]]:
        """
        Get list of available actions in this environment.
        
        Returns:
            List of action definitions
        """
        return [
            {
                "type": "function",
                "name": "execute",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "type": "function",
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "type": "function",
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern",
                        },
                    },
                },
            },
        ]


class LocalEnvironment(BaseEnvironment):
    """
    Local environment that executes commands on the host system.
    
    This environment is suitable for development and testing,
    but should not be used for untrusted code.
    """
    
    async def setup(self) -> None:
        """Set up the local environment."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._setup_complete = True
    
    async def teardown(self) -> None:
        """Clean up the local environment."""
        # Optionally cleanup workspace
        pass
    
    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> EnvironmentResponse:
        """
        Execute a command locally.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            EnvironmentResponse with execution results
        """
        import time
        start = time.time()
        
        if timeout is None:
            timeout = self.config.timeout
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            return EnvironmentResponse(
                success=result.returncode == 0,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                execution_time=time.time() - start,
            )
        except subprocess.TimeoutExpired:
            return EnvironmentResponse(
                success=False,
                error="Command timed out",
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )


def create_environment(
    config: EnvironmentConfig,
) -> BaseEnvironment:
    """
    Factory function to create an environment instance.
    
    Args:
        config: Environment configuration
        
    Returns:
        Configured environment instance
    """
    return LocalEnvironment(config)
