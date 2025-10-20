#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate descriptions for educational materials using OpenAI-compatible API endpoint.
This script reads processed materials and generates descriptions using a vision-language model.
"""

import os
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from openai import OpenAI
from config_loader import load_config


# Configuration is now loaded via dotenv at module level


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
            api_base: Base URL for the OpenAI-compatible API (defaults from config.env)
            api_key: API key (defaults from config.env, use "EMPTY" for local vLLM servers)
            model_name: Name of the model to use (defaults from config.env)
            timeout: Request timeout in seconds (defaults from config.env)
        """
        # Read configuration from YAML
        cfg = load_config()
        # Get values from YAML or use provided values or fallback defaults
        if api_base is None:
            api_base = (cfg.get('api', {}).get('base_url') or '').strip()
            if not api_base:
                server_host = cfg.get('server', {}).get('host', '0.0.0.0')
                server_port = cfg.get('server', {}).get('port', 8000)
                api_base = f"http://{server_host}:{server_port}/v1"

        if api_key is None:
            api_key = cfg.get('api', {}).get('key', 'EMPTY')

        if model_name is None:
            model_name = cfg.get('model', {}).get('name', 'Qwen/Qwen3-VL-30B-A3B-Instruct')

        if timeout is None:
            timeout = int(cfg.get('api', {}).get('timeout', 3600))
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )
        self.model_name = model_name
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_message_with_image(
        self, 
        image_path: str, 
        prompt: str,
        use_url: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create a message with image for the API.
        
        Args:
            image_path: Path to the image file or URL
            prompt: Text prompt to send with the image
            use_url: If True, treat image_path as URL; otherwise encode as base64
            
        Returns:
            List of message dictionaries
        """
        if use_url:
            # Use URL directly
            image_url = image_path
        else:
            # Encode image as base64
            base64_image = self.encode_image_to_base64(image_path)
            image_url = f"data:image/jpeg;base64,{base64_image}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        return messages
    
    def create_message_with_multiple_images(
        self,
        image_paths: List[str],
        prompt: str,
        use_url: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create a message with multiple images for the API.
        
        Args:
            image_paths: List of paths to image files or URLs
            prompt: Text prompt to send with the images
            use_url: If True, treat image_paths as URLs; otherwise encode as base64
            
        Returns:
            List of message dictionaries
        """
        content = []
        
        # Add all images
        for image_path in image_paths:
            if use_url:
                image_url = image_path
            else:
                base64_image = self.encode_image_to_base64(image_path)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        
        # Add the prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
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
        guided_json: Optional[Dict[str, Any]] = None
    ) -> str | Dict[str, Any]:
        """
        Generate a description for an image using streaming with usage tracking.
        
        Args:
            image_path: Path to the image file or URL
            prompt: Text prompt describing what to generate
            temperature: Sampling temperature (0.0 to 2.0, defaults from config.env)
            top_p: Nucleus sampling parameter (defaults from config.env)
            top_k: Top-k sampling parameter (defaults from config.env)
            presence_penalty: Presence penalty (-2.0 to 2.0, defaults from config.env)
            max_tokens: Maximum number of tokens to generate (defaults from config.env)
            use_url: If True, treat image_path as URL
            verbose: If True, print timing information
            return_metrics: If True, return dict with content and metrics
            
        Returns:
            Generated description text, or dict with 'content' and 'metrics' if return_metrics=True
        """
        # Use config values if not provided
        cfg = load_config()
        if temperature is None:
            temperature = float(cfg.get('generation', {}).get('temperature', 0.6))
        if top_p is None:
            top_p = float(cfg.get('generation', {}).get('top_p', 0.95))
        if top_k is None:
            top_k = int(cfg.get('generation', {}).get('top_k', 20))
        if presence_penalty is None:
            presence_penalty = float(cfg.get('generation', {}).get('presence_penalty', 0.0))
        if max_tokens is None:
            max_tokens = int(cfg.get('generation', {}).get('max_tokens', 4096))
        
        messages = self.create_message_with_image(image_path, prompt, use_url=use_url)
        
        try:
            start_time = time.time()
            extra_body = {"top_k": top_k}
            if guided_json is not None:
                extra_body["guided_json"] = guided_json
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body=extra_body,
                stream=True,
                stream_options={"include_usage": True}
            )
            
            content = ""
            first_token_time = None
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            for chunk in stream:
                # Track time to first token
                if first_token_time is None and chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content:
                        first_token_time = time.time()
                
                # Collect content
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content
                
                # Extract usage information (typically in the last chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens
            
            end_time = time.time()
            total_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
            
            # Calculate throughput metrics
            metrics = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'total_time': total_time,
                'ttft': ttft,
                'generation_time': generation_time,
                'pp_per_sec': prompt_tokens / ttft if ttft > 0 else 0,
                'tg_per_sec': completion_tokens / generation_time if generation_time > 0 else 0,
            }
            
            if return_metrics:
                return {
                    'content': content,
                    'metrics': metrics
                }
            
            return content
        
        except Exception as e:
            print(f"Error generating description: {e}")
            raise
    
    def process_materials_db(
        self,
        db_path: Optional[str] = None,
        materials_dir: Optional[str] = None,
        prompt_template: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Process all materials from SQLite database and generate descriptions.
        Handles both individual page rows and batch "all" rows.
        
        Args:
            db_path: Path to the SQLite database file (defaults from config.env)
            materials_dir: Directory containing the processed material images (defaults from config.env)
            prompt_template: Custom prompt template (uses default or from config.env if None)
            batch_size: Number of images to process in each request (defaults from config.env)
        """
        # Use config values if not provided
        cfg = load_config()
        if db_path is None:
            db_path = cfg.get('paths', {}).get('materials_db_path', 'materials/processed_materials.db')

        if materials_dir is None:
            materials_dir = cfg.get('paths', {}).get('materials_dir', 'materials/processed')

        if batch_size is None:
            batch_size = int(cfg.get('batch', {}).get('size', 10))
        
        # Load prompt templates:
        # 1) Explicit function arg overrides everything
        # 2) Otherwise prefer PROMPT_TEMPLATE_SINGLE for per-page and PROMPT_TEMPLATE_BATCH for batch
        # 3) Fallback to legacy PROMPT_TEMPLATE, then to built-in defaults
        legacy_prompt = None
        single_prompt_default = (
            "Return only a JSON object with fields exactly: needed (boolean), key_concept (string), description (string). "
            "Rules: needed=true only if the page teaches a substantive concept, method, worked example, or definition. "
            "Mark needed=false for pages that are: title/cover, course schedule/timeline, syllabus outline, table of contents, announcements, logistics (deadlines, office hours, emails), exam/quiz information, instructions to reflect/prepare, grading/policies, decorative/quote/blank pages, single-panel comics or cartoons, inspirational quotes, and high-level overview/agenda slides that summarize sections without teaching details. "
            "key_concept is a 2-6 word phrase naming the primary concept taught on this page; if multiple topics appear, choose the most central. "
            "description is 1-2 short sentences defining the concept and what a student needs to know to answer multiple-choice questions about it; keep under 40 words; plain text; no lists; ignore headers/footers and long OCR passages. Output JSON only with no extra text."
        )
        batch_prompt_default = (
            "Return only a JSON object with fields exactly: key_concept (array of 5 strings), description (string). Do not include a needed field. "
            "The selected pages will already exclude administrative content (title, schedule, announcements, logistics, exam info, syllabus/TOC). "
            "Rules: key_concept must contain exactly five phrases (each 2-6 words) capturing the main concepts covered across the selected pages. "
            "description is 3-4 sentences summarizing the overall topic, key ideas, and typical problem types or skills assessed; concise, <=100 words; plain text; no bullets; avoid page-level details. Output JSON only with no extra text."
        )
        cfg = load_config()
        single_prompt_from_env = cfg.get('prompts', {}).get('single') or legacy_prompt or single_prompt_default
        batch_prompt_from_env = cfg.get('prompts', {}).get('batch') or legacy_prompt or batch_prompt_default
        # If caller provided a prompt_template, use it for both modes (explicit override)
        if prompt_template is not None:
            single_prompt_from_env = prompt_template
            batch_prompt_from_env = prompt_template
        
        # Connect to database
        db_path_obj = Path(db_path)
        materials_dir_obj = Path(materials_dir)
        
        if not db_path_obj.exists():
            print(f"Error: Database file not found at {db_path}")
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Ensure schema has needed columns (idempotent migration)
        try:
            cursor.execute('PRAGMA table_info(materials)')
            cols = [r[1] for r in cursor.fetchall()]
            if 'needed' not in cols:
                cursor.execute('ALTER TABLE materials ADD COLUMN needed INTEGER')
            if 'key_concept' not in cols:
                cursor.execute('ALTER TABLE materials ADD COLUMN key_concept TEXT')
            conn.commit()
        except Exception:
            pass
        
        # Read all materials from database
        cursor.execute('SELECT id, original_filename, current_filename, status, description, needed, key_concept FROM materials')
        rows = []
        for row in cursor.fetchall():
            rows.append({
                'id': row[0],
                'original_filename': row[1],
                'current_filename': row[2],
                'status': row[3] or '',
                'description': row[4] or '',
                'needed': row[5] if row[5] is not None else None,
                'key_concept': row[6] or ''
            })
        
        print(f"Found {len(rows)} materials to process")
        
        # Collect metrics
        all_metrics = []
        
        # First pass: identify all "all" rows to build batch_rows mapping
        batch_rows = {}  # Map of original_filename -> {'all_row_idx': idx, 'page_indices': []}
        
        for idx, row in enumerate(rows):
            if row['current_filename'] == 'all':
                original_filename = row['original_filename']
                # Skip if already processed
                if row.get('status') == 'processed' and row.get('description'):
                    continue
                batch_rows[original_filename] = {'all_row_idx': idx, 'page_indices': []}
        
        # Second pass: identify individual page rows and link them to batch rows
        rows_to_process = []
        
        for idx, row in enumerate(rows):
            # Skip if already processed
            if row.get('status') == 'processed' and row.get('description'):
                continue
            
            current_filename = row['current_filename']
            original_filename = row['original_filename']
            
            # Skip "all" rows in this pass - we'll handle them separately
            if current_filename == 'all':
                continue
            
            # This is an individual page row
            image_path = materials_dir_obj / current_filename
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                row['status'] = 'error'
                row['description'] = 'Image file not found'
                continue
            
            rows_to_process.append((idx, row, image_path, 'individual'))
            
            # Track this page for batch processing if an "all" row exists
            if original_filename in batch_rows:
                batch_rows[original_filename]['page_indices'].append(idx)
        
        # Third pass: prepare batch rows for processing
        for original_filename, batch_info in batch_rows.items():
            all_row_idx = batch_info['all_row_idx']
            page_indices = batch_info['page_indices']
            
            if page_indices:
                # Defer filtering based on 'needed' until after individual processing
                rows_to_process.append((all_row_idx, rows[all_row_idx], page_indices, 'batch'))
        
        print(f"Need to process {len(rows_to_process)} tasks (including batch tasks)")
        
        # Process materials
        print("\n" + "="*80)
        print("Processing Materials")
        print("="*80)
        
        for task_idx, (row_idx, row, data, task_type) in enumerate(rows_to_process):
            if task_type == 'individual':
                image_path = data
                print(f"\n[{task_idx+1}/{len(rows_to_process)}] Processing (individual): {row['current_filename']}")
                
                try:
                    # Structured output schema for single page
                    single_schema = {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "needed": {"type": "boolean"},
                            "key_concept": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["needed", "key_concept", "description"]
                    }
                    result = self.generate_description(
                        str(image_path),
                        single_prompt_from_env,
                        use_url=False,
                        return_metrics=True,
                        verbose=True,
                        guided_json=single_schema
                    )
                    
                    # Parse structured content
                    content_text = result['content']
                    try:
                        parsed = json.loads(content_text)
                    except Exception as parse_err:
                        raise ValueError(f"Failed to parse JSON: {parse_err}; content: {content_text}")
                    description = parsed.get('description', '')
                    key_concept = parsed.get('key_concept', '')
                    needed_val = parsed.get('needed', None)
                    metrics = result['metrics']
                    
                    all_metrics.append(metrics)
                    
                    row['status'] = 'processed'
                    row['description'] = description
                    row['key_concept'] = key_concept
                    # Normalize needed to 0/1/None
                    if isinstance(needed_val, bool):
                        row['needed'] = 1 if needed_val else 0
                    elif needed_val in (0, 1):
                        row['needed'] = int(needed_val)
                    else:
                        row['needed'] = None
                    
                    print(f"  ✓ PP: {metrics['prompt_tokens']}, TG: {metrics['completion_tokens']}")
                    print(f"  ✓ TTFT: {metrics['ttft']:.3f}s, TG/sec: {metrics['tg_per_sec']:.2f}, PP/sec: {metrics['pp_per_sec']:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    row['status'] = 'error'
                    row['description'] = str(e)
                    row['needed'] = None
                    row['key_concept'] = row.get('key_concept', '')
            
            elif task_type == 'batch':
                page_indices = data
                original_filename = row['original_filename']
                # Filter pages by needed == true (1)
                filtered_image_paths = []
                for page_idx in page_indices:
                    page_row = rows[page_idx]
                    if page_row.get('needed') in (1, True):
                        img_path = materials_dir_obj / page_row['current_filename']
                        if img_path.exists():
                            filtered_image_paths.append(str(img_path))
                print(f"\n[{task_idx+1}/{len(rows_to_process)}] Processing (batch, {len(filtered_image_paths)} images): {original_filename}")
                
                try:
                    if not filtered_image_paths:
                        row['status'] = 'processed'
                        row['description'] = ''
                        row['key_concept'] = json.dumps([])
                        row['needed'] = None
                        print("  ✓ No relevant pages selected (needed==false for all).")
                        continue
                    # Create message with multiple images
                    messages = self.create_message_with_multiple_images(
                        filtered_image_paths,
                        batch_prompt_from_env,
                        use_url=False
                    )
                    
                    # Generate description for batch
                    start_time = time.time()
                    cfg = load_config()
                    extra_body = {
                        "top_k": int(cfg.get('generation', {}).get('top_k', 20)),
                        "guided_json": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "key_concept": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 5,
                                    "maxItems": 5
                                },
                                "description": {"type": "string"}
                            },
                            "required": ["key_concept", "description"]
                        }
                    }
                    stream = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=float(cfg.get('generation', {}).get('temperature', 0.6)),
                        top_p=float(cfg.get('generation', {}).get('top_p', 0.95)),
                        presence_penalty=float(cfg.get('generation', {}).get('presence_penalty', 0.0)),
                        max_tokens=int(cfg.get('generation', {}).get('max_tokens', 4096)),
                        extra_body=extra_body,
                        stream=True,
                        stream_options={"include_usage": True}
                    )
                    
                    content = ""
                    first_token_time = None
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    
                    for chunk in stream:
                        # Track time to first token
                        if first_token_time is None and chunk.choices and len(chunk.choices) > 0:
                            if chunk.choices[0].delta.content:
                                first_token_time = time.time()
                        
                        # Collect content
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                content += delta.content
                        
                        # Extract usage information
                        if hasattr(chunk, 'usage') and chunk.usage:
                            prompt_tokens = chunk.usage.prompt_tokens
                            completion_tokens = chunk.usage.completion_tokens
                            total_tokens = chunk.usage.total_tokens
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    ttft = first_token_time - start_time if first_token_time else 0
                    generation_time = end_time - first_token_time if first_token_time else total_time
                    
                    # Calculate metrics
                    metrics = {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens,
                        'total_time': total_time,
                        'ttft': ttft,
                        'generation_time': generation_time,
                        'pp_per_sec': prompt_tokens / ttft if ttft > 0 else 0,
                        'tg_per_sec': completion_tokens / generation_time if generation_time > 0 else 0,
                    }
                    
                    all_metrics.append(metrics)
                    
                    # Parse structured JSON
                    try:
                        parsed = json.loads(content)
                    except Exception as parse_err:
                        raise ValueError(f"Failed to parse JSON: {parse_err}; content: {content}")
                    row['status'] = 'processed'
                    row['description'] = parsed.get('description', '')
                    row['key_concept'] = json.dumps(parsed.get('key_concept', []))
                    row['needed'] = None
                    
                    print(f"  ✓ Batch processed: {len(filtered_image_paths)} images")
                    print(f"  ✓ PP: {metrics['prompt_tokens']}, TG: {metrics['completion_tokens']}")
                    print(f"  ✓ TTFT: {metrics['ttft']:.3f}s, TG/sec: {metrics['tg_per_sec']:.2f}, PP/sec: {metrics['pp_per_sec']:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    row['status'] = 'error'
                    row['description'] = str(e)
                    row['key_concept'] = json.dumps([])
                    row['needed'] = None
        
        # Update database with results
        for row in rows:
            needed_val = row.get('needed')
            if isinstance(needed_val, bool):
                needed_val = 1 if needed_val else 0
            cursor.execute('''
                UPDATE materials 
                SET status = ?, description = ?, needed = ?, key_concept = ?
                WHERE id = ?
            ''', (row.get('status', ''), row.get('description', ''), needed_val, row.get('key_concept', ''), row['id']))
        
        conn.commit()
        conn.close()
        
        print(f"\n✓ Updated database saved to: {db_path}")
        
        # Print summary
        processed = sum(1 for row in rows if row['status'] == 'processed')
        errors = sum(1 for row in rows if row['status'] == 'error')
        print(f"\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total materials: {len(rows)}")
        print(f"Successfully processed: {processed}")
        print(f"Errors: {errors}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: 1 (single inference)")
        
        # Print benchmark comparison if both modes were run
        if all_metrics:
            print(f"\n" + "="*80)
            print("PERFORMANCE METRICS")
            print("="*80)
            
            total_time = sum(m['total_time'] for m in all_metrics)
            total_gen_time = sum(m['generation_time'] for m in all_metrics)
            avg_ttft = sum(m['ttft'] for m in all_metrics) / len(all_metrics)
            avg_tg_per_sec = sum(m['tg_per_sec'] for m in all_metrics) / len(all_metrics)
            avg_pp_per_sec = sum(m['pp_per_sec'] for m in all_metrics) / len(all_metrics)
            total_pp = sum(m['prompt_tokens'] for m in all_metrics)
            total_tg = sum(m['completion_tokens'] for m in all_metrics)
            
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


def main():
    """Main function to demonstrate usage."""
    # Initialize generator with OpenAI-compatible endpoint using YAML config
    generator = MaterialsDescriptionGenerator()
    
    # # Example 1: Single image with URL and metrics reporting
    # print("=" * 80)
    # print("Example 1: Generate description with metrics")
    # print("=" * 80)
    # result = generator.generate_description(
    #     "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
    #     "Read all the text in the image.",
    #     use_url=True,
    #     return_metrics=True
    # )
    
    # Example 2: Process all materials from database (params loaded from YAML)
    generator.process_materials_db()


if __name__ == "__main__":
    main()

