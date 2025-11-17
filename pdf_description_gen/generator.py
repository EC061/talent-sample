#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main generator class for educational materials description generation.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from openai import OpenAI

from config_loader import load_config
from .api_client import APIClient
from .database import DatabaseManager
from .logger import MarkdownLogger
from .schemas import get_single_page_schema, get_batch_schema, prepare_structured_output
from .prompts import get_prompts_from_config
from .types import (
    MaterialRow, PerformanceMetrics, GenerationResult,
    BatchInfo, ProcessingTask, TaskType
)
from .utils import extract_page_number


class MaterialsDescriptionGenerator:
    """Generate descriptions for educational materials using OpenAI-compatible API."""
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize the description generator.
        
        Args:
            api_base: Base URL for the OpenAI-compatible API (defaults from config.yml)
            api_key: API key (defaults from config.yml)
            model_name: Name of the model to use (defaults from config.yml)
            timeout: Request timeout in seconds (defaults from config.yml)
        """
        # Read configuration from YAML
        self.cfg = load_config()
        self.platform = self.cfg.get('api', {}).get('platform', 'openai').lower()
        
        # Get platform-specific API configuration
        api_config = self.cfg.get('api', {}).get(self.platform, {})
        
        # Service tier
        service_tier = api_config.get('service_tier')
        if isinstance(service_tier, str):
            service_tier = service_tier.strip().lower()
        self.service_tier: Optional[str] = service_tier or None
        
        # API parameters
        if api_base is None:
            api_base = (api_config.get('base_url') or '').strip()
            if not api_base:
                api_base = 'https://api.openai.com/v1'
        
        if api_key is None:
            api_key = api_config.get('key', '')
        
        if model_name is None:
            model_name = api_config.get('vlm_model', 'gpt-4o')
        
        if timeout is None:
            timeout = int(self.cfg.get('api', {}).get('timeout', 3600))
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )
        
        self.model_name = model_name
        self.api_client = APIClient(client, model_name, self.service_tier)
        self.logger = MarkdownLogger()
    
    def generate_description(
        self,
        image_path: str,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_url: bool = False,
        verbose: bool = False,
        return_metrics: bool = False,
        guided_json: Optional[Dict[str, Any]] = None,
        image_number: Optional[int] = None
    ) -> str | GenerationResult:
        """
        Generate a description for an image using streaming with usage tracking.
        
        Args:
            image_path: Path to the image file or URL
            prompt: Text prompt describing what to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (unused, kept for compatibility)
            top_k: Top-k sampling parameter (unused, kept for compatibility)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            use_url: If True, treat image_path as URL
            verbose: If True, print timing information
            return_metrics: If True, return GenerationResult with content and metrics
            guided_json: Optional JSON schema for structured output
            image_number: Optional image number for log filename
            
        Returns:
            Generated description text, or GenerationResult if return_metrics=True
        """
        messages = APIClient.create_message_with_image(image_path, prompt, use_url=use_url)
        
        response_format = None
        if guided_json is not None:
            _, response_format = prepare_structured_output(guided_json)
        
        content, metrics = self.api_client.generate_with_retry(
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            verbose=verbose
        )
        
        # Log to markdown file
        self.logger.log_request(
            request_type="single_page",
            prompt=prompt,
            content=content,
            metrics=metrics,
            model_name=self.model_name,
            image_path=image_path,
            image_number=image_number
        )
        
        if return_metrics:
            return GenerationResult(content=content, metrics=metrics)
        
        return content
    
    def _build_batch_rows(self, rows: List[MaterialRow]) -> Dict[str, BatchInfo]:
        """
        Build mapping of batch rows (rows with current_filename == 'all').
        
        Args:
            rows: List of all material rows
            
        Returns:
            Dict mapping original_filename to BatchInfo
        """
        batch_rows: Dict[str, BatchInfo] = {}
        
        for idx, row in enumerate(rows):
            if row.current_filename == 'all':
                # Skip if already processed
                if row.status == 'processed' and row.description:
                    continue
                batch_rows[row.original_filename] = BatchInfo(
                    all_row_idx=idx,
                    page_indices=[]
                )
        
        return batch_rows
    
    def _build_processing_tasks(
        self,
        rows: List[MaterialRow],
        batch_rows: Dict[str, BatchInfo],
        materials_dir: Path
    ) -> List[ProcessingTask]:
        """
        Build list of processing tasks (individual pages and batches).
        
        Args:
            rows: List of all material rows
            batch_rows: Mapping of batch rows
            materials_dir: Directory containing material images
            
        Returns:
            List of ProcessingTask objects
        """
        tasks: List[ProcessingTask] = []
        
        # Process individual page rows
        for idx, row in enumerate(rows):
            # Skip if already processed
            if row.status == 'processed' and row.description:
                continue
            
            current_filename = row.current_filename
            original_filename = row.original_filename
            
            # Skip "all" rows in this pass
            if current_filename == 'all':
                continue
            
            # This is an individual page row
            image_path = materials_dir / current_filename
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                row.status = 'error'
                row.description = 'Image file not found'
                continue
            
            tasks.append(ProcessingTask(
                row_idx=idx,
                row=row,
                data=str(image_path),
                task_type=TaskType.INDIVIDUAL
            ))
            
            # Track this page for batch processing if an "all" row exists
            if original_filename in batch_rows:
                batch_rows[original_filename].page_indices.append(idx)
        
        # Add batch tasks
        for original_filename, batch_info in batch_rows.items():
            all_row_idx = batch_info.all_row_idx
            page_indices = batch_info.page_indices
            
            if page_indices:
                tasks.append(ProcessingTask(
                    row_idx=all_row_idx,
                    row=rows[all_row_idx],
                    data=page_indices,
                    task_type=TaskType.BATCH
                ))
        
        return tasks
    
    def _process_individual_task(
        self,
        task: ProcessingTask,
        single_prompt: str,
        task_idx: int,
        total_tasks: int,
        all_metrics: List[PerformanceMetrics]
    ) -> None:
        """
        Process an individual page task.
        
        Args:
            task: Processing task
            single_prompt: Prompt template for single page
            task_idx: Current task index (for progress display)
            total_tasks: Total number of tasks
            all_metrics: List to append metrics to
        """
        image_path = task.data
        row = task.row
        
        print(f"\n[{task_idx+1}/{total_tasks}] Processing (individual): {row.current_filename}")
        
        # Extract page number from filename for logging
        page_number = extract_page_number(row.current_filename)
        
        try:
            result = self.generate_description(
                image_path,
                single_prompt,
                use_url=False,
                return_metrics=True,
                verbose=True,
                guided_json=get_single_page_schema(),
                image_number=page_number
            )
            
            # Parse structured content
            content_text = result.content if isinstance(result, GenerationResult) else result
            try:
                parsed = json.loads(content_text)
            except Exception as parse_err:
                raise ValueError(f"Failed to parse JSON: {parse_err}; content: {content_text}")
            
            description = parsed.get('description', '')
            key_concept = parsed.get('key_concept', '')
            needed_val = parsed.get('needed', None)
            
            metrics = result.metrics if isinstance(result, GenerationResult) else None
            if metrics:
                all_metrics.append(metrics)
            
            row.status = 'processed'
            row.description = description
            row.key_concept = key_concept
            
            # Normalize needed to 0/1/None
            if isinstance(needed_val, bool):
                row.needed = 1 if needed_val else 0
            elif needed_val in (0, 1):
                row.needed = int(needed_val)
            else:
                row.needed = None
            
            if metrics:
                print(f"  ✓ PP: {metrics.prompt_tokens}, TG: {metrics.completion_tokens}")
                print(f"  ✓ TTFT: {metrics.ttft:.3f}s, TG/sec: {metrics.tg_per_sec:.2f}, PP/sec: {metrics.pp_per_sec:.2f}")
                if metrics.total_cost is not None:
                    print(f"  ✓ Cost: ${metrics.total_cost:.6f} (Input: ${metrics.input_cost:.6f}, Output: ${metrics.output_cost:.6f})")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            row.status = 'error'
            row.description = str(e)
            row.needed = None
            row.key_concept = row.key_concept or ''
    
    def _process_batch_task(
        self,
        task: ProcessingTask,
        rows: List[MaterialRow],
        batch_prompt: str,
        materials_dir: Path,
        task_idx: int,
        total_tasks: int,
        all_metrics: List[PerformanceMetrics]
    ) -> None:
        """
        Process a batch task.
        
        Args:
            task: Processing task
            rows: List of all material rows
            batch_prompt: Prompt template for batch
            materials_dir: Directory containing material images
            task_idx: Current task index (for progress display)
            total_tasks: Total number of tasks
            all_metrics: List to append metrics to
        """
        page_indices = task.data
        row = task.row
        original_filename = row.original_filename
        
        # Filter pages by needed == true (1)
        filtered_image_paths: List[str] = []
        page_numbers: List[int] = []
        
        for page_idx in page_indices:
            page_row = rows[page_idx]
            if page_row.needed in (1, True):
                img_path = materials_dir / page_row.current_filename
                if img_path.exists():
                    filtered_image_paths.append(str(img_path))
                    # Extract page number for logging
                    page_num = extract_page_number(page_row.current_filename)
                    if page_num is not None:
                        page_numbers.append(page_num)
        
        print(f"\n[{task_idx+1}/{total_tasks}] Processing (batch, {len(filtered_image_paths)} images): {original_filename}")
        
        # Use first page number for batch log filename, or None if no pages
        batch_page_number = page_numbers[0] if page_numbers else None
        
        try:
            if not filtered_image_paths:
                row.status = 'processed'
                row.description = ''
                row.key_concept = json.dumps([])
                row.needed = None
                print("  ✓ No relevant pages selected (needed==false for all).")
                return
            
            # Create message with multiple images
            messages = APIClient.create_message_with_multiple_images(
                filtered_image_paths,
                batch_prompt,
                use_url=False
            )
            
            # Generate description for batch
            _, response_format = prepare_structured_output(get_batch_schema())
            
            content, metrics = self.api_client.generate_with_retry(
                messages=messages,
                response_format=response_format,
                verbose=True
            )
            
            all_metrics.append(metrics)
            
            # Parse structured JSON
            try:
                parsed = json.loads(content)
            except Exception as parse_err:
                raise ValueError(f"Failed to parse JSON: {parse_err}; content: {content}")
            
            row.status = 'processed'
            row.description = parsed.get('description', '')
            row.key_concept = json.dumps(parsed.get('key_concept', []))
            row.needed = None
            
            # Log to markdown file
            self.logger.log_request(
                request_type="batch",
                prompt=batch_prompt,
                content=content,
                metrics=metrics,
                model_name=self.model_name,
                image_path=f"{len(filtered_image_paths)} images",
                additional_info={
                    "Original Filename": original_filename,
                    "Number of Images": len(filtered_image_paths),
                    "Page Numbers": page_numbers if page_numbers else "N/A"
                },
                image_number=batch_page_number
            )
            
            print(f"  ✓ Batch processed: {len(filtered_image_paths)} images")
            print(f"  ✓ PP: {metrics.prompt_tokens}, TG: {metrics.completion_tokens}")
            print(f"  ✓ TTFT: {metrics.ttft:.3f}s, TG/sec: {metrics.tg_per_sec:.2f}, PP/sec: {metrics.pp_per_sec:.2f}")
            if metrics.total_cost is not None:
                print(f"  ✓ Cost: ${metrics.total_cost:.6f} (Input: ${metrics.input_cost:.6f}, Output: ${metrics.output_cost:.6f})")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            row.status = 'error'
            row.description = str(e)
            row.key_concept = json.dumps([])
            row.needed = None
    
    def process_materials_db(
        self,
        db_path: Optional[str] = None,
        materials_dir: Optional[str] = None,
        prompt_template: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Process all materials from SQLite database and generate descriptions.
        Handles both individual page rows and batch "all" rows.
        
        Args:
            db_path: Path to the SQLite database file (defaults from config.yml)
            materials_dir: Directory containing the processed material images (defaults from config.yml)
            prompt_template: Custom prompt template (uses default or from config.yml if None)
            batch_size: Number of images to process in each request (defaults from config.yml)
        """
        # Use config values if not provided
        if db_path is None:
            db_path = self.cfg.get('paths', {}).get('materials_db_path', 'materials/processed_materials.db')
        
        if materials_dir is None:
            materials_dir = self.cfg.get('paths', {}).get('materials_dir', 'materials/processed')
        
        if batch_size is None:
            batch_size = int(self.cfg.get('batch', {}).get('size', 10))
        
        # Load prompt templates
        single_prompt, batch_prompt = get_prompts_from_config(self.cfg)
        if prompt_template is not None:
            single_prompt = prompt_template
            batch_prompt = prompt_template
        
        materials_dir_obj = Path(materials_dir)
        
        # Process materials with database manager
        with DatabaseManager(db_path) as db:
            rows = db.fetch_all_materials()
            print(f"Found {len(rows)} materials to process")
            
            # Build batch rows and processing tasks
            batch_rows = self._build_batch_rows(rows)
            tasks = self._build_processing_tasks(rows, batch_rows, materials_dir_obj)
            
            print(f"Need to process {len(tasks)} tasks (including batch tasks)")
            
            # Collect metrics
            all_metrics: List[PerformanceMetrics] = []
            
            # Process materials
            print("\n" + "="*80)
            print("Processing Materials")
            print("="*80)
            
            for task_idx, task in enumerate(tasks):
                if task.task_type == TaskType.INDIVIDUAL:
                    self._process_individual_task(
                        task, single_prompt, task_idx, len(tasks), all_metrics
                    )
                elif task.task_type == TaskType.BATCH:
                    self._process_batch_task(
                        task, rows, batch_prompt, materials_dir_obj,
                        task_idx, len(tasks), all_metrics
                    )
            
            # Update database with results
            db.update_materials_batch(rows)
            db.commit()
        
        print(f"\n✓ Updated database saved to: {db_path}")
        
        # Print summary
        self._print_summary(rows, batch_size, all_metrics)
    
    def _print_summary(
        self,
        rows: List[MaterialRow],
        batch_size: int,
        all_metrics: List[PerformanceMetrics]
    ) -> None:
        """
        Print processing summary and performance metrics.
        
        Args:
            rows: List of all material rows
            batch_size: Configured batch size
            all_metrics: List of performance metrics
        """
        processed = sum(1 for row in rows if row.status == 'processed')
        errors = sum(1 for row in rows if row.status == 'error')
        
        print(f"\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total materials: {len(rows)}")
        print(f"Successfully processed: {processed}")
        print(f"Errors: {errors}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: 1 (single inference)")
        
        # Print benchmark comparison if metrics are available
        if all_metrics:
            print(f"\n" + "="*80)
            print("PERFORMANCE METRICS")
            print("="*80)
            
            total_time = sum(m.total_time for m in all_metrics)
            total_gen_time = sum(m.generation_time for m in all_metrics)
            avg_ttft = sum(m.ttft for m in all_metrics) / len(all_metrics)
            avg_tg_per_sec = sum(m.tg_per_sec for m in all_metrics) / len(all_metrics)
            avg_pp_per_sec = sum(m.pp_per_sec for m in all_metrics) / len(all_metrics)
            total_pp = sum(m.prompt_tokens for m in all_metrics)
            total_tg = sum(m.completion_tokens for m in all_metrics)
            total_input_cost = sum(m.input_cost or 0 for m in all_metrics)
            total_output_cost = sum(m.output_cost or 0 for m in all_metrics)
            total_cost = sum(m.total_cost or 0 for m in all_metrics)
            
            print(f"\n{'Metric':<30}    ")
            print("-" * 55)
            print(f"{'Total requests':<30} {len(all_metrics)}")
            print(f"{'Total PP tokens':<30} {total_pp}")
            print(f"{'Total TG tokens':<30} {total_tg}")
            print(f"{'Total time (s)':<30} {total_time:.2f}")
            print(f"{'Total generation time (s)':<30} {total_gen_time:.2f}")
            print(f"{'Avg TTFT (s)':<30} {avg_ttft:.3f}")
            print(f"{'Avg TG/sec (tokens/s)':<30} {avg_tg_per_sec:.2f}")
            print(f"{'Avg PP/sec (tokens/s)':<30} {avg_pp_per_sec:.2f}")
            if total_cost > 0:
                print(f"{'Total cost ($)':<30} {total_cost:.6f}")
                print(f"{'Total input cost ($)':<30} {total_input_cost:.6f}")
                print(f"{'Total output cost ($)':<30} {total_output_cost:.6f}")

