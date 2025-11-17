#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Description Generation Package

A well-structured package for generating descriptions of educational materials
using OpenAI-compatible vision-language models.
"""

from .generator import MaterialsDescriptionGenerator
from .types import (
    MaterialRow,
    PerformanceMetrics,
    GenerationResult,
    TaskType,
    BatchInfo,
    ProcessingTask
)
from .schemas import (
    get_single_page_schema,
    get_batch_schema,
    prepare_structured_output
)
from .api_client import APIClient
from .database import DatabaseManager
from .logger import MarkdownLogger
from .prompts import (
    SINGLE_PAGE_PROMPT_DEFAULT,
    BATCH_PROMPT_DEFAULT,
    get_prompts_from_config
)
from .utils import extract_page_number

__version__ = "2.0.0"

__all__ = [
    # Main class
    "MaterialsDescriptionGenerator",
    
    # Types
    "MaterialRow",
    "PerformanceMetrics",
    "GenerationResult",
    "TaskType",
    "BatchInfo",
    "ProcessingTask",
    
    # Schemas
    "get_single_page_schema",
    "get_batch_schema",
    "prepare_structured_output",
    
    # Components
    "APIClient",
    "DatabaseManager",
    "MarkdownLogger",
    
    # Prompts
    "SINGLE_PAGE_PROMPT_DEFAULT",
    "BATCH_PROMPT_DEFAULT",
    "get_prompts_from_config",
    
    # Utils
    "extract_page_number",
]

