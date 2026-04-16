"""
OptiS Benchmark - Parser Module

This module provides parsing utilities for various file formats
used in optical design.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class ParsedLens:
    """Represents a parsed lens system."""
    name: str
    surfaces: int
    wavelengths: list[dict[str, float]]
    fields: list[dict[str, Any]]
    aperture: dict[str, Any]
    metadata: dict[str, Any]


class JSONLParser:
    """Parser for JSON Lines format."""
    
    @staticmethod
    def read(
        file_path: Union[str, Path],
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Read JSONL file.
        
        Args:
            file_path: Path to JSONL file
            limit: Maximum number of lines to read
            
        Returns:
            List of parsed JSON objects
        """
        results = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON at line {i+1}: {e}")
        
        return results
    
    @staticmethod
    def write(
        file_path: Union[str, Path],
        records: list[dict[str, Any]],
        append: bool = False,
    ) -> None:
        """
        Write records to JSONL file.
        
        Args:
            file_path: Path to output file
            records: List of records to write
            append: If True, append to existing file
        """
        mode = "a" if append else "w"
        
        with open(file_path, mode, encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class YAMLParser:
    """Parser for YAML configuration files."""
    
    @staticmethod
    def read(file_path: Union[str, Path]) -> dict[str, Any]:
        """Read YAML file."""
        import yaml
        
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def write(
        file_path: Union[str, Path],
        data: dict[str, Any],
    ) -> None:
        """Write data to YAML file."""
        import yaml
        
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


class ConfigParser:
    """Parser for OptiS configuration files."""
    
    # Pattern to match environment variable references
    ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")
    
    @classmethod
    def expand_env_vars(cls, value: Any) -> Any:
        """
        Recursively expand environment variables in a value.
        
        Supports ${VAR_NAME} syntax.
        """
        import os
        
        if isinstance(value, str):
            def replace_env(match):
                var_name = match.group(1)
                return os.environ.get(var_name, "")
            return cls.ENV_VAR_PATTERN.sub(replace_env, value)
        elif isinstance(value, dict):
            return {k: cls.expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.expand_env_vars(item) for item in value]
        return value
    
    @classmethod
    def load_config(cls, file_path: Union[str, Path]) -> dict[str, Any]:
        """
        Load configuration file with environment variable expansion.
        
        Supports both JSON and YAML files.
        """
        path = Path(file_path)
        
        if path.suffix in (".yaml", ".yml"):
            data = YAMLParser.read(path)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls.expand_env_vars(data)


class ResultsParser:
    """Parser for evaluation results."""
    
    @staticmethod
    def load_results(
        file_path: Union[str, Path],
        as_jsonl: bool = True,
    ) -> dict[str, Any]:
        """
        Load evaluation results.
        
        Args:
            file_path: Path to results file
            as_jsonl: If True, treat as JSONL format
            
        Returns:
            Parsed results
        """
        path = Path(file_path)
        
        if as_jsonl or path.suffix == ".jsonl":
            return {
                "results": JSONLParser.read(path),
            }
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    
    @staticmethod
    def format_results_summary(results: dict[str, Any]) -> str:
        """
        Format results into a human-readable summary.
        
        Args:
            results: Parsed results
            
        Returns:
            Formatted summary string
        """
        output = []
        output.append("=" * 60)
        output.append("EVALUATION RESULTS SUMMARY")
        output.append("=" * 60)
        
        if "timestamp" in results:
            output.append(f"Timestamp: {results['timestamp']}")
        if "task_id" in results:
            output.append(f"Task: {results['task_id']}")
        if "model" in results:
            output.append(f"Model: {results['model']}")
        
        output.append("")
        
        # Statistics
        total = results.get("total_tasks", 0)
        successful = results.get("successful_tasks", 0)
        rate = results.get("success_rate", 0.0) * 100
        avg_score = results.get("avg_score", 0.0) * 100
        total_cost = results.get("total_cost", 0.0)
        avg_time = results.get("avg_execution_time", 0.0)
        
        output.append(f"Total Tasks:     {total}")
        output.append(f"Successful:       {successful} ({rate:.1f}%)")
        output.append(f"Average Score:   {avg_score:.1f}%")
        output.append(f"Total Cost:      ${total_cost:.4f}")
        output.append(f"Avg Time/Task:   {avg_time:.1f}s")
        
        # Metrics summary
        metrics = results.get("metrics_summary", {})
        if metrics:
            output.append("")
            output.append("METRICS SUMMARY")
            output.append("-" * 40)
            
            for name, values in metrics.items():
                output.append(f"\n{name}:")
                output.append(f"  Mean: {values.get('mean', 0):.4f}")
                output.append(f"  Min:  {values.get('min', 0):.4f}")
                output.append(f"  Max:  {values.get('max', 0):.4f}")
                output.append(f"  Std:  {values.get('std', 0):.4f}")
        
        output.append("")
        output.append("=" * 60)
        
        return "\n".join(output)


class OpticalDataParser:
    """Parser for optical design data files."""
    
    @staticmethod
    def parse_zemax_lens(file_path: Union[str, Path]) -> ParsedLens:
        """
        Parse a Zemax lens file (.zmx).
        
        This is a simplified parser - full Zemax format parsing
        would require the ZOS-API or Zemax DLL.
        """
        # Simplified parsing - real implementation would use ZOS-API
        return ParsedLens(
            name="Unknown Lens",
            surfaces=0,
            wavelengths=[],
            fields=[],
            aperture={},
            metadata={},
        )
    
    @staticmethod
    def parse_mtf_data(content: str) -> dict[str, Any]:
        """
        Parse MTF (Modulation Transfer Function) data.
        
        Args:
            content: MTF data string
            
        Returns:
            Parsed MTF data
        """
        data = {
            "spatial_frequency": [],
            "mtf_tangential": [],
            "mtf_sagittal": [],
        }
        
        # Simple parsing - extract numeric values
        lines = content.strip().split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    freq = float(parts[0])
                    tang = float(parts[1])
                    sag = float(parts[2])
                    
                    data["spatial_frequency"].append(freq)
                    data["mtf_tangential"].append(tang)
                    data["mtf_sagittal"].append(sag)
                except ValueError:
                    continue
        
        return data
    
    @staticmethod
    def parse_spot_data(content: str) -> dict[str, Any]:
        """
        Parse spot diagram data.
        
        Args:
            content: Spot data string
            
        Returns:
            Parsed spot data
        """
        data = {
            "rays": [],
            "rms_radius": 0.0,
            "diffraction_limit": 0.0,
        }
        
        # Parse ray positions
        lines = content.strip().split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    data["rays"].append({"x": x, "y": y})
                except ValueError:
                    continue
        
        return data
