#!/usr/bin/env python3
"""
Lightweight YAML configuration loader with sensible defaults.

Can also be run as a CLI tool to print environment variable exports for shell consumption:

Usage:
  eval "$(python3 config_loader.py [vlm|llm])"
  
Args:
  model_type: Either 'vlm' or 'llm' (default: 'vlm')
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Fetch nested key via dot.path with default."""
    cur: Any = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load YAML config. If not provided, searches `config.yml` in project root.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / 'config.yml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def as_env_dict(cfg: Dict[str, Any], model_type: str = 'vlm') -> Dict[str, str]:
    """
    Flatten a subset of settings to the historical ENV variable names
    expected by existing scripts. Used to ease migration with bash.
    
    Args:
        cfg: The configuration dictionary
        model_type: Either 'vlm' or 'llm' to specify which model config to use
    """
    env: Dict[str, str] = {}

    # Model/server - choose the right model based on type
    if model_type == 'llm':
        env['MODEL_NAME'] = str(_deep_get(cfg, 'model.llm.name', 'QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ'))
    else:
        env['MODEL_NAME'] = str(_deep_get(cfg, 'model.vlm.name', 'Qwen/Qwen3-VL-30B-A3B-Instruct'))
    
    env['SERVER_HOST'] = str(_deep_get(cfg, 'server.host', '0.0.0.0'))
    env['SERVER_PORT'] = str(_deep_get(cfg, 'server.port', 8000))

    return env


def main() -> None:
    """CLI entry point for printing environment variable exports."""
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'vlm'
    if model_type not in ('vlm', 'llm'):
        print(f"Error: Invalid model type '{model_type}'. Must be 'vlm' or 'llm'.", file=sys.stderr)
        sys.exit(1)
    
    cfg = load_config()
    env = as_env_dict(cfg, model_type=model_type)
    for key, value in env.items():
        if value is None:
            continue
        # Safely quote values for shell
        sval = str(value).replace('"', '\\"')
        print(f'export {key}="{sval}"')


if __name__ == "__main__":
    main()


