#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging functionality for API requests and responses.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .types import PerformanceMetrics


class MarkdownLogger:
    """Logs API requests and responses to markdown files."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize markdown logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def log_request(
        self,
        request_type: str,
        prompt: str,
        content: str,
        metrics: PerformanceMetrics,
        model_name: str,
        image_path: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        image_number: Optional[int] = None
    ) -> None:
        """
        Log API request and response to a markdown file.
        
        Args:
            request_type: Type of request (e.g., "single_page", "batch")
            prompt: The prompt sent to the API
            content: The response content
            metrics: Performance metrics
            model_name: Name of the model used
            image_path: Optional path to the image(s) used
            additional_info: Optional dictionary with additional information
            image_number: Optional image number for filename (falls back to timestamp)
        """
        if image_number is not None:
            log_file = self.log_dir / f"pdf_description_{image_number:05d}.md"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file = self.log_dir / f"pdf_description_{timestamp}.md"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("# PDF Description Generation Log\n\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Request Type:** {request_type}\n")
            f.write(f"**Model:** {model_name}\n\n")
            
            # Request details
            f.write("## Request Details\n\n")
            if image_path:
                f.write(f"**Image Path:** `{image_path}`\n\n")
            
            f.write("### Prompt\n\n")
            f.write("```\n")
            f.write(prompt)
            f.write("\n```\n\n")
            
            if additional_info:
                f.write("### Additional Information\n\n")
                for key, value in additional_info.items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            # Response
            f.write("## Response\n\n")
            f.write("### Content\n\n")
            f.write("```json\n")
            try:
                parsed = json.loads(content)
                f.write(json.dumps(parsed, indent=2, ensure_ascii=False))
            except:
                f.write(content)
            f.write("\n```\n\n")
            
            # Metrics table
            self._write_metrics_table(f, metrics)
    
    def _write_metrics_table(self, f, metrics: PerformanceMetrics) -> None:
        """
        Write metrics table to file.
        
        Args:
            f: File handle
            metrics: Performance metrics to write
        """
        f.write("## Token Usage & Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Prompt Tokens | {metrics.prompt_tokens} |\n")
        f.write(f"| Completion Tokens | {metrics.completion_tokens} |\n")
        f.write(f"| Total Tokens | {metrics.total_tokens} |\n")
        f.write(f"| Total Time (s) | {metrics.total_time:.3f} |\n")
        f.write(f"| Time to First Token (s) | {metrics.ttft:.3f} |\n")
        f.write(f"| Generation Time (s) | {metrics.generation_time:.3f} |\n")
        f.write(f"| Prompt Processing Rate (tokens/s) | {metrics.pp_per_sec:.2f} |\n")
        f.write(f"| Token Generation Rate (tokens/s) | {metrics.tg_per_sec:.2f} |\n")
        
        if metrics.total_cost is not None:
            f.write(f"| **Total Cost ($)** | **{metrics.total_cost:.6f}** |\n")
            f.write(f"| Input Cost ($) | {metrics.input_cost:.6f} |\n")
            f.write(f"| Output Cost ($) | {metrics.output_cost:.6f} |\n")
        
        f.write("\n")

