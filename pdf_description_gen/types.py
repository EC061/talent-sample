#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type definitions and data classes for PDF description generation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class TaskType(Enum):
    """Type of processing task."""
    INDIVIDUAL = "individual"
    BATCH = "batch"


@dataclass
class MaterialRow:
    """Represents a row in the materials database."""
    id: int
    original_filename: str
    current_filename: str
    status: str
    description: str
    needed: Optional[int]
    key_concept: str
    
    @classmethod
    def from_db_row(cls, row: tuple) -> 'MaterialRow':
        """Create MaterialRow from database row tuple."""
        return cls(
            id=row[0],
            original_filename=row[1],
            current_filename=row[2],
            status=row[3] or '',
            description=row[4] or '',
            needed=row[5] if row[5] is not None else None,
            key_concept=row[6] or ''
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for API requests."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_time: float
    ttft: float  # Time to first token
    generation_time: float
    pp_per_sec: float  # Prompt processing rate
    tg_per_sec: float  # Token generation rate
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    total_cost: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'total_time': self.total_time,
            'ttft': self.ttft,
            'generation_time': self.generation_time,
            'pp_per_sec': self.pp_per_sec,
            'tg_per_sec': self.tg_per_sec,
        }
        if self.input_cost is not None:
            result['input_cost'] = self.input_cost
        if self.output_cost is not None:
            result['output_cost'] = self.output_cost
        if self.total_cost is not None:
            result['total_cost'] = self.total_cost
        return result


@dataclass
class GenerationResult:
    """Result of a description generation request."""
    content: str
    metrics: PerformanceMetrics


@dataclass
class BatchInfo:
    """Information about a batch processing task."""
    all_row_idx: int
    page_indices: list[int]


@dataclass
class ProcessingTask:
    """Represents a processing task (individual or batch)."""
    row_idx: int
    row: MaterialRow
    data: Any  # Either image_path (str) for individual or page_indices (list) for batch
    task_type: TaskType

