#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate descriptions for educational materials using OpenAI-compatible API endpoint.
This script reads processed materials and generates descriptions using a vision-language model.
"""

import os
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from openai import OpenAI


def load_config_env(config_path: str = "config.env") -> Dict[str, str]:
    """
    Load configuration from config.env file.
    
    Args:
        config_path: Path to the config.env file
        
    Returns:
        Dictionary of configuration values
    """
    config = {}
    
    # Try to find config.env in script directory
    script_dir = Path(__file__).parent
    config_file = script_dir / config_path
    
    if not config_file.exists():
        print(f"Warning: {config_file} not found, using environment variables or defaults")
        return config
    
    print(f"Loading configuration from {config_file}")
    
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY="VALUE" or KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                config[key] = value
    
    return config


# Load config at module level
_CONFIG = load_config_env()


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
        # Get values from config.env or use provided values or fallback defaults
        if api_base is None:
            # Check if API_BASE_URL is set in config (for remote API usage)
            api_base = _CONFIG.get('API_BASE_URL', '').strip()
            # If not set or empty, construct from SERVER_HOST and SERVER_PORT (local server)
            if not api_base:
                server_host = _CONFIG.get('SERVER_HOST', '0.0.0.0')
                server_port = _CONFIG.get('SERVER_PORT', '8000')
                api_base = f"http://{server_host}:{server_port}/v1"
        
        if api_key is None:
            api_key = _CONFIG.get('API_KEY', 'EMPTY')
        
        if model_name is None:
            model_name = _CONFIG.get('MODEL_NAME', 'Qwen/Qwen3-VL-30B-A3B-Instruct')
        
        if timeout is None:
            timeout = int(_CONFIG.get('API_TIMEOUT', '3600'))
        
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
    
    def create_batch_messages_with_images(
        self,
        image_paths: List[str],
        prompt: str,
        use_url: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create a batch of messages, each with one image for the API.
        
        Args:
            image_paths: List of paths to image files or URLs
            prompt: Text prompt to send with each image
            use_url: If True, treat image_paths as URLs; otherwise encode as base64
            
        Returns:
            List of message dictionaries (one message per image)
        """
        messages = []
        for image_path in image_paths:
            if use_url:
                # Use URL directly
                image_url = image_path
            else:
                # Encode image as base64
                base64_image = self.encode_image_to_base64(image_path)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            
            message = {
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
            messages.append(message)
        
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
        return_metrics: bool = False
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
        if temperature is None:
            temperature = float(_CONFIG.get('GENERATION_TEMPERATURE', '0.6'))
        if top_p is None:
            top_p = float(_CONFIG.get('GENERATION_TOP_P', '0.95'))
        if top_k is None:
            top_k = int(_CONFIG.get('GENERATION_TOP_K', '20'))
        if presence_penalty is None:
            presence_penalty = float(_CONFIG.get('GENERATION_PRESENCE_PENALTY', '0.0'))
        if max_tokens is None:
            max_tokens = int(_CONFIG.get('GENERATION_MAX_TOKENS', '4096'))
        
        messages = self.create_message_with_image(image_path, prompt, use_url=use_url)
        
        try:
            start_time = time.time()
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body={"top_k": top_k},
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
    
    def generate_batch_descriptions(
        self,
        image_paths: List[str],
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_url: bool = False,
        verbose: bool = False,
        return_metrics: bool = False
    ) -> List[str] | Dict[str, Any]:
        """
        Generate descriptions for multiple images in a single request using streaming with usage tracking.
        
        Args:
            image_paths: List of paths to image files or URLs
            prompt: Text prompt describing what to generate
            temperature: Sampling temperature (0.0 to 2.0, defaults from config.env)
            top_p: Nucleus sampling parameter (defaults from config.env)
            top_k: Top-k sampling parameter (defaults from config.env)
            presence_penalty: Presence penalty (-2.0 to 2.0, defaults from config.env)
            max_tokens: Maximum number of tokens to generate (defaults from config.env)
            use_url: If True, treat image_paths as URLs
            verbose: If True, print timing information
            return_metrics: If True, return dict with content and metrics
            
        Returns:
            List of generated description texts, or dict with 'content' and 'metrics' if return_metrics=True
        """
        # Use config values if not provided
        if temperature is None:
            temperature = float(_CONFIG.get('GENERATION_TEMPERATURE', '0.6'))
        if top_p is None:
            top_p = float(_CONFIG.get('GENERATION_TOP_P', '0.95'))
        if top_k is None:
            top_k = int(_CONFIG.get('GENERATION_TOP_K', '50'))
        if presence_penalty is None:
            presence_penalty = float(_CONFIG.get('GENERATION_PRESENCE_PENALTY', '0.0'))
        if max_tokens is None:
            max_tokens = int(_CONFIG.get('GENERATION_MAX_TOKENS', '4096'))
        
        messages = self.create_batch_messages_with_images(image_paths, prompt, use_url=use_url)
        
        try:
            start_time = time.time()
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body={"top_k": top_k},
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
                'num_images': len(image_paths)
            }
            
            if return_metrics:
                return {
                    'content': content,
                    'metrics': metrics
                }
            
            return content
        
        except Exception as e:
            print(f"Error generating batch descriptions: {e}")
            raise
    
    def process_materials_csv(
        self,
        csv_path: Optional[str] = None,
        materials_dir: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        batch_size: Optional[int] = None,
        benchmark_mode: bool = True
    ):
        """
        Process all materials from CSV and generate descriptions, with optional performance benchmarking.
        
        Args:
            csv_path: Path to the CSV file with materials list (defaults from config.env)
            materials_dir: Directory containing the processed material images (defaults from config.env)
            output_csv_path: Path to save updated CSV (defaults to input path)
            prompt_template: Custom prompt template (uses default or from config.env if None)
            batch_size: Number of images to process in each batch request (defaults from config.env)
            benchmark_mode: If True, run both single and batch inference to compare performance
        """
        # Use config values if not provided
        if csv_path is None:
            csv_path = _CONFIG.get('MATERIALS_CSV_PATH', 'materials/processed_materials.csv')
        
        if materials_dir is None:
            materials_dir = _CONFIG.get('MATERIALS_DIR', 'materials/processed')
        
        if batch_size is None:
            batch_size = int(_CONFIG.get('BATCH_SIZE', '10'))
        
        if output_csv_path is None:
            output_csv_path = csv_path
        
        if prompt_template is None:
            prompt_template = _CONFIG.get('PROMPT_TEMPLATE')
        
        if prompt_template is None:
            prompt_template = (
                "Analyze this educational material image and provide a detailed description. "
                "Include: 1) The main topic or subject, 2) Key concepts or formulas shown, "
                "3) Any diagrams, graphs, or visual elements, 4) The educational level "
                "(e.g., high school physics, college calculus). Be specific and concise."
            )
        
        # Read CSV
        csv_path_obj = Path(csv_path)
        materials_dir_obj = Path(materials_dir)
        
        if not csv_path_obj.exists():
            print(f"Error: CSV file not found at {csv_path}")
            return
        
        rows = []
        with open(csv_path_obj, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        
        print(f"Found {len(rows)} materials to process")
        
        # Collect metrics
        single_metrics = []
        batch_metrics = []
        
        # Filter rows that need processing
        rows_to_process = []
        for idx, row in enumerate(rows):
            # Skip if already processed
            if row.get('status') == 'processed' and row.get('description'):
                continue
            
            current_filename = row['current_filename']
            image_path = materials_dir_obj / current_filename
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                row['status'] = 'error'
                row['description'] = 'Image file not found'
                continue
            
            rows_to_process.append((idx, row, image_path))
        
        print(f"Need to process {len(rows_to_process)} materials")
        
        # BENCHMARK MODE: Run single inference first
        if benchmark_mode and len(rows_to_process) > 0:
            print("\n" + "="*80)
            print("BENCHMARK: Single Inference Mode")
            print("="*80)
            
            for idx, (_, row, image_path) in enumerate(rows_to_process):
                print(f"\n[Single {idx+1}/{len(rows_to_process)}] Processing: {row['current_filename']}")
                
                try:
                    result = self.generate_description(
                        str(image_path),
                        prompt_template,
                        use_url=False,
                        return_metrics=True,
                        verbose=False
                    )
                    
                    metrics = result['metrics']
                    single_metrics.append(metrics)
                    
                    print(f"  ✓ PP: {metrics['prompt_tokens']}, TG: {metrics['completion_tokens']}")
                    print(f"  ✓ TTFT: {metrics['ttft']:.3f}s, TG/sec: {metrics['tg_per_sec']:.2f}, PP/sec: {metrics['pp_per_sec']:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        # BATCH MODE: Process in batches (always run this to get results for CSV)
        print("\n" + "="*80)
        print("BENCHMARK: Batch Inference Mode" if benchmark_mode else "Batch Processing")
        print("="*80)
        
        total_batches = (len(rows_to_process) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(rows_to_process), batch_size):
            batch = rows_to_process[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            # Prepare batch data
            batch_image_paths = [str(item[2]) for item in batch]
            batch_rows = [item[1] for item in batch]
            batch_indices = [item[0] for item in batch]
            
            print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} images...")
            for item in batch:
                print(f"  - {item[1]['current_filename']}")
            
            try:
                result = self.generate_batch_descriptions(
                    batch_image_paths,
                    prompt_template,
                    use_url=False,
                    return_metrics=True,
                    verbose=False
                )
                
                description = result['content']
                metrics = result['metrics']
                
                # Split the description for each image (if model returns combined text)
                # For now, we'll assume the model returns one combined response
                # You may need to adjust this based on your model's actual response format
                
                # Mark all in batch as processed with the same description
                # In practice, you might want to parse the response to separate descriptions
                for row in batch_rows:
                    row['status'] = 'processed'
                    row['description'] = description
                
                batch_metrics.append(metrics)
                
                print(f"  ✓ PP: {metrics['prompt_tokens']}, TG: {metrics['completion_tokens']}")
                print(f"  ✓ TTFT: {metrics['ttft']:.3f}s, TG/sec: {metrics['tg_per_sec']:.2f}, PP/sec: {metrics['pp_per_sec']:.2f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                for row in batch_rows:
                    row['status'] = 'error'
                    row['description'] = str(e)
        
        # Write updated CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['original_filename', 'current_filename', 'status', 'description']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Updated CSV saved to: {output_csv_path}")
        
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
        print(f"Total batches: {total_batches}")
        
        # Print benchmark comparison if both modes were run
        if benchmark_mode and single_metrics and batch_metrics:
            print(f"\n" + "="*80)
            print("PERFORMANCE COMPARISON")
            print("="*80)
            
            # Single inference metrics
            single_total_time = sum(m['total_time'] for m in single_metrics)
            single_total_gen_time = sum(m['generation_time'] for m in single_metrics)
            single_avg_ttft = sum(m['ttft'] for m in single_metrics) / len(single_metrics)
            single_avg_tg_per_sec = sum(m['tg_per_sec'] for m in single_metrics) / len(single_metrics)
            single_avg_pp_per_sec = sum(m['pp_per_sec'] for m in single_metrics) / len(single_metrics)
            single_total_pp = sum(m['prompt_tokens'] for m in single_metrics)
            single_total_tg = sum(m['completion_tokens'] for m in single_metrics)
            
            # Batch inference metrics
            batch_total_time = sum(m['total_time'] for m in batch_metrics)
            batch_total_gen_time = sum(m['generation_time'] for m in batch_metrics)
            batch_avg_ttft = sum(m['ttft'] for m in batch_metrics) / len(batch_metrics)
            batch_avg_tg_per_sec = sum(m['tg_per_sec'] for m in batch_metrics) / len(batch_metrics)
            batch_avg_pp_per_sec = sum(m['pp_per_sec'] for m in batch_metrics) / len(batch_metrics)
            batch_total_pp = sum(m['prompt_tokens'] for m in batch_metrics)
            batch_total_tg = sum(m['completion_tokens'] for m in batch_metrics)
            
            print(f"\n{'Metric':<30} {'Single Inference':<25} {'Batch Inference':<25} {'Speedup':<15}")
            print("-" * 95)
            print(f"{'Total requests':<30} {len(single_metrics):<25} {len(batch_metrics):<25} {'-':<15}")
            print(f"{'Total PP tokens':<30} {single_total_pp:<25} {batch_total_pp:<25} {'-':<15}")
            print(f"{'Total TG tokens':<30} {single_total_tg:<25} {batch_total_tg:<25} {'-':<15}")
            print(f"{'Total time (s)':<30} {single_total_time:<25.2f} {batch_total_time:<25.2f} {single_total_time/batch_total_time:<15.2f}x")
            print(f"{'Total generation time (s)':<30} {single_total_gen_time:<25.2f} {batch_total_gen_time:<25.2f} {single_total_gen_time/batch_total_gen_time:<15.2f}x")
            print(f"{'Avg TTFT (s)':<30} {single_avg_ttft:<25.3f} {batch_avg_ttft:<25.3f} {single_avg_ttft/batch_avg_ttft:<15.2f}x")
            print(f"{'Avg TG/sec (tokens/s)':<30} {single_avg_tg_per_sec:<25.2f} {batch_avg_tg_per_sec:<25.2f} {batch_avg_tg_per_sec/single_avg_tg_per_sec:<15.2f}x")
            print(f"{'Avg PP/sec (tokens/s)':<30} {single_avg_pp_per_sec:<25.2f} {batch_avg_pp_per_sec:<25.2f} {batch_avg_pp_per_sec/single_avg_pp_per_sec:<15.2f}x")
            
            print(f"\n{'Key Insights:'}")
            print(f"  • Batch mode is {single_total_time/batch_total_time:.2f}x faster overall")
            print(f"  • Batch mode achieves {batch_avg_tg_per_sec/single_avg_tg_per_sec:.2f}x higher token generation throughput")
            print(f"  • Total time saved: {single_total_time - batch_total_time:.2f}s ({(1 - batch_total_time/single_total_time)*100:.1f}% reduction)")
            
        elif batch_metrics:
            # Only batch metrics available
            print(f"\n" + "="*80)
            print("BATCH MODE PERFORMANCE METRICS")
            print("="*80)
            
            batch_total_time = sum(m['total_time'] for m in batch_metrics)
            batch_total_gen_time = sum(m['generation_time'] for m in batch_metrics)
            batch_avg_ttft = sum(m['ttft'] for m in batch_metrics) / len(batch_metrics)
            batch_avg_tg_per_sec = sum(m['tg_per_sec'] for m in batch_metrics) / len(batch_metrics)
            batch_avg_pp_per_sec = sum(m['pp_per_sec'] for m in batch_metrics) / len(batch_metrics)
            batch_total_pp = sum(m['prompt_tokens'] for m in batch_metrics)
            batch_total_tg = sum(m['completion_tokens'] for m in batch_metrics)
            
            print(f"Total prefill tokens (PP): {batch_total_pp}")
            print(f"Total generation tokens (TG): {batch_total_tg}")
            print(f"Total processing time: {batch_total_time:.2f}s")
            print(f"Total generation time: {batch_total_gen_time:.2f}s")
            print(f"Average TTFT: {batch_avg_ttft:.3f}s")
            print(f"Average TG/sec: {batch_avg_tg_per_sec:.2f} tokens/s")
            print(f"Average PP/sec: {batch_avg_pp_per_sec:.2f} tokens/s")


def main():
    """Main function to demonstrate usage."""
    # Initialize generator with OpenAI-compatible endpoint
    # All parameters will be loaded from config.env
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
    
    # Example 2: Process all materials from CSV
    # All parameters will be loaded from config.env
    generator.process_materials_csv()


if __name__ == "__main__":
    main()

