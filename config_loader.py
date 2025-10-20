#!/usr/bin/env python3
"""
Lightweight YAML configuration loader with sensible defaults.
"""

import os
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


def as_env_dict(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Flatten a subset of settings to the historical ENV variable names
    expected by existing scripts. Used to ease migration with bash.
    """
    env: Dict[str, str] = {}

    # Model/server
    env['MODEL_NAME'] = str(_deep_get(cfg, 'model.name', 'Qwen/Qwen3-VL-30B-A3B-Instruct'))
    env['SERVER_HOST'] = str(_deep_get(cfg, 'server.host', '0.0.0.0'))
    env['SERVER_PORT'] = str(_deep_get(cfg, 'server.port', 8000))

    # vLLM
    tp_size = _deep_get(cfg, 'vllm.tensor_parallel_size', None)
    if tp_size is not None and str(tp_size).strip() != '':
        env['TENSOR_PARALLEL_SIZE'] = str(tp_size)
    cuda_devices = _deep_get(cfg, 'vllm.cuda_devices', None)
    if cuda_devices is not None and str(cuda_devices).strip() != '':
        env['CUDA_DEVICES'] = str(cuda_devices)
    env['VLLM_WORKER_MULTIPROC_METHOD'] = str(_deep_get(cfg, 'vllm.worker_multiproc_method', 'spawn'))
    env['VLLM_USE_FLASHINFER_MOE_FP16'] = '1' if bool(_deep_get(cfg, 'vllm.use_flashinfer_moe_fp16', True)) else '0'
    env['PYTORCH_CUDA_ALLOC_CONF'] = str(_deep_get(cfg, 'vllm.pytorch_cuda_alloc_conf', 'expandable_segments:True'))
    env['VLLM_MAX_MODEL_LEN'] = str(_deep_get(cfg, 'vllm.max_model_len', 32768))
    env['VLLM_MAX_NUM_SEQS'] = str(_deep_get(cfg, 'vllm.max_num_seqs', 8))
    env['VLLM_GPU_MEMORY_UTILIZATION'] = str(_deep_get(cfg, 'vllm.gpu_memory_utilization', 0.80))
    env['VLLM_SEED'] = str(_deep_get(cfg, 'vllm.seed', 0))
    env['VLLM_MM_ENCODER_TP_MODE'] = str(_deep_get(cfg, 'vllm.mm_encoder_tp_mode', 'data'))
    env['VLLM_ENABLE_EXPERT_PARALLEL'] = 'true' if bool(_deep_get(cfg, 'vllm.enable_expert_parallel', True)) else 'false'

    # API
    env['API_BASE_URL'] = str(_deep_get(cfg, 'api.base_url', ''))
    env['API_KEY'] = str(_deep_get(cfg, 'api.key', 'EMPTY'))
    env['API_TIMEOUT'] = str(_deep_get(cfg, 'api.timeout', 3600))

    # Generation
    env['GENERATION_TEMPERATURE'] = str(_deep_get(cfg, 'generation.temperature', 0.6))
    env['GENERATION_TOP_P'] = str(_deep_get(cfg, 'generation.top_p', 0.95))
    env['GENERATION_TOP_K'] = str(_deep_get(cfg, 'generation.top_k', 20))
    env['GENERATION_PRESENCE_PENALTY'] = str(_deep_get(cfg, 'generation.presence_penalty', 0.0))
    env['GENERATION_MAX_TOKENS'] = str(_deep_get(cfg, 'generation.max_tokens', 4096))

    # Batch
    env['BATCH_SIZE'] = str(_deep_get(cfg, 'batch.size', 10))

    # Paths
    env['MATERIALS_DB_PATH'] = str(_deep_get(cfg, 'paths.materials_db_path', 'materials/processed_materials.db'))
    env['MATERIALS_DIR'] = str(_deep_get(cfg, 'paths.materials_dir', 'materials/processed'))
    env['PDF_DIR'] = str(_deep_get(cfg, 'paths.pdf_dir', 'materials/pdf'))

    # Prompts
    env['PROMPT_TEMPLATE_SINGLE'] = str(_deep_get(cfg, 'prompts.single', ''))
    env['PROMPT_TEMPLATE_BATCH'] = str(_deep_get(cfg, 'prompts.batch', ''))
    return env


