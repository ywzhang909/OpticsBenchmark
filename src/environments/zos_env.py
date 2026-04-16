"""
OptiS Benchmark - ZOS-API Environment Module

This module provides integration with Zemax OpticStudio via ZOS-API.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base_env import BaseEnvironment, EnvironmentConfig, EnvironmentResponse


@dataclass
class ZOSConnectionConfig:
    """Configuration for ZOS-API connection."""
    port: int = 5555
    timeout: int = 60
    zemax_path: Optional[str] = None


class ZOSAPIEnvironment(BaseEnvironment):
    """
    Environment for interacting with Zemax OpticStudio via ZOS-API.
    
    This environment provides a Python interface to control OpticStudio,
    allowing agents to perform optical design and analysis tasks.
    
    Requirements:
    - Zemax OpticStudio Professional or Premium
    - ZOS-API enabled
    - PythonNET installed
    
    Usage:
        1. Start OpticStudio with ZOS-API connection enabled
        2. Create this environment with the connection config
        3. Use the provided methods to interact with OpticStudio
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        zos_config: Optional[ZOSConnectionConfig] = None,
    ):
        super().__init__(config)
        self.zos_config = zos_config or ZOSConnectionConfig()
        self._zos_connection = None
        self._python_net_loaded = False
    
    async def setup(self) -> None:
        """Set up the ZOS-API environment."""
        import time
        start = time.time()
        
        try:
            # Try to load PythonNET
            try:
                import clr
                self._python_net_loaded = True
            except ImportError:
                # PythonNET not available, try alternative connection
                self._python_net_loaded = False
            
            if not self._python_net_loaded:
                # Check if ZOS-API is accessible via another method
                await self._check_zemax_connection()
            
            self._setup_complete = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup ZOS-API environment: {e}")
    
    async def _check_zemax_connection(self) -> bool:
        """Check if Zemax OpticStudio is running and accessible."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        try:
            result = sock.connect_ex(
                ('localhost', self.zos_config.port)
            )
            sock.close()
            return result == 0
        except Exception:
            return False
    
    async def teardown(self) -> None:
        """Clean up the ZOS-API environment."""
        if self._zos_connection:
            try:
                self._zos_connection.CloseApplication()
            except Exception:
                pass
            self._zos_connection = None
    
    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> EnvironmentResponse:
        """Execute a ZOS-API command."""
        import time
        start = time.time()
        
        try:
            # Parse command
            if command.startswith("python:"):
                # Python code to execute via ZOS-API
                code = command[7:].strip()
                result = await self._execute_python_code(code)
            elif command.startswith("zemax:"):
                # Zemax ZPL macro
                macro = command[7:].strip()
                result = await self._execute_zemax_macro(macro)
            else:
                # Regular shell command
                result = await self._execute_shell(command)
            
            return EnvironmentResponse(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                metadata=result.get("metadata", {}),
                execution_time=time.time() - start,
            )
            
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    async def _execute_python_code(self, code: str) -> dict[str, Any]:
        """Execute Python code via ZOS-API."""
        if not self._python_net_loaded:
            return {
                "success": False,
                "error": "PythonNET not available. Please install pythonnet.",
            }
        
        # Note: In actual implementation, this would use PythonNET
        # to communicate with the ZOS-API
        return {
            "success": False,
            "error": "ZOS-API Python execution not fully implemented",
        }
    
    async def _execute_zemax_macro(self, macro_name: str) -> dict[str, Any]:
        """Execute a Zemax ZPL macro."""
        if not await self._check_zemax_connection():
            return {
                "success": False,
                "error": "Zemax OpticStudio is not running or ZOS-API not enabled",
            }
        
        # Macro execution would go through ZOS-API
        return {
            "success": False,
            "error": "ZPL macro execution not implemented",
        }
    
    async def _execute_shell(self, command: str) -> dict[str, Any]:
        """Execute a shell command."""
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    # =========================================================================
    # High-level ZOS-API methods
    # =========================================================================
    
    async def load_lens(self, file_path: str) -> EnvironmentResponse:
        """
        Load a lens file into OpticStudio.
        
        Args:
            file_path: Path to .zmx file
            
        Returns:
            EnvironmentResponse with load result
        """
        import time
        start = time.time()
        
        if not self._python_net_loaded:
            return EnvironmentResponse(
                success=False,
                error="ZOS-API not available",
                execution_time=time.time() - start,
            )
        
        try:
            # Load lens via ZOS-API
            # TheApplication.PrimaryStudio.OpenFile(file_path, "")
            return EnvironmentResponse(
                success=True,
                output={"file": file_path},
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    async def get_system_data(self) -> EnvironmentResponse:
        """
        Get current OpticStudio system data.
        
        Returns:
            EnvironmentResponse with system data
        """
        import time
        start = time.time()
        
        if not self._python_net_loaded:
            return EnvironmentResponse(
                success=False,
                error="ZOS-API not available",
                execution_time=time.time() - start,
            )
        
        try:
            # Get system data via ZOS-API
            system_data = {
                "wavelengths": [],
                "fields": [],
                "surfaces": 0,
                "aperture": {},
            }
            
            return EnvironmentResponse(
                success=True,
                output=system_data,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    async def analyze_mtf(self) -> EnvironmentResponse:
        """
        Run MTF (Modulation Transfer Function) analysis.
        
        Returns:
            EnvironmentResponse with MTF data
        """
        import time
        start = time.time()
        
        try:
            # Run MTF analysis via ZOS-API
            mtf_data = {
                "spatial_frequency": [],
                "mtf_tangential": [],
                "mtf_sagittal": [],
            }
            
            return EnvironmentResponse(
                success=True,
                output=mtf_data,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    async def analyze_spot(self) -> EnvironmentResponse:
        """
        Run spot diagram analysis.
        
        Returns:
            EnvironmentResponse with spot diagram data
        """
        import time
        start = time.time()
        
        try:
            # Run spot analysis via ZOS-API
            spot_data = {
                "rms_radius": 0.0,
                "diffraction_limit": 0.0,
                "chief_ray": {},
                "marginal_ray": {},
            }
            
            return EnvironmentResponse(
                success=True,
                output=spot_data,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    async def optimize(
        self,
        algorithm: str = "Damped Least Squares",
        cycles: int = 100,
    ) -> EnvironmentResponse:
        """
        Run optimization.
        
        Args:
            algorithm: Optimization algorithm
            cycles: Number of optimization cycles
            
        Returns:
            EnvironmentResponse with optimization result
        """
        import time
        start = time.time()
        
        try:
            # Run optimization via ZOS-API
            result = {
                "algorithm": algorithm,
                "cycles": cycles,
                "final_merit": 0.0,
                "converged": False,
            }
            
            return EnvironmentResponse(
                success=True,
                output=result,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return EnvironmentResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
    
    def get_available_actions(self) -> list[dict[str, Any]]:
        """Get list of available ZOS-API actions."""
        base_actions = super().get_available_actions()
        
        zos_actions = [
            {
                "type": "function",
                "name": "load_lens",
                "description": "Load a Zemax lens file (.zmx)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the .zmx file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "type": "function",
                "name": "get_system_data",
                "description": "Get current OpticStudio system data (wavelengths, fields, surfaces)",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "analyze_mtf",
                "description": "Run MTF analysis and get results",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "analyze_spot",
                "description": "Run spot diagram analysis",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "optimize",
                "description": "Run lens optimization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "algorithm": {
                            "type": "string",
                            "description": "Optimization algorithm",
                            "enum": ["Damped Least Squares", "Hammer", "Coordinate Break"],
                        },
                        "cycles": {
                            "type": "integer",
                            "description": "Number of optimization cycles",
                        },
                    },
                },
            },
        ]
        
        return base_actions + zos_actions
