#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-step recommendation system for educational materials.
Step 1: Select the most relevant file based on question and wrong answer.
Step 2: Select the most relevant page range within the selected file.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from config_loader import load_config


class MaterialRecommendationSystem:
    """Two-step recommendation system for educational materials using LLM."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize the recommendation system.

        Args:
            api_base: Base URL for the OpenAI-compatible API (defaults from config.yml)
            api_key: API key (defaults from config.yml, use "EMPTY" for local servers)
            model_name: Name of the LLM model to use (defaults from config.yml)
            timeout: Request timeout in seconds (defaults from config.yml)
        """
        # Read configuration from YAML
        cfg = load_config()
        self.platform = cfg.get('api', {}).get('platform', 'vllm').lower()
        self.cfg = cfg
        
        # Get platform-specific API configuration
        api_config = cfg.get('api', {}).get(self.platform, {})

        # Determine optional service tier for OpenAI platform
        service_tier = api_config.get("service_tier") if self.platform == 'openai' else None
        if isinstance(service_tier, str):
            service_tier = service_tier.strip().lower()
        self.service_tier = service_tier or None

        # Get values from YAML or use provided values or fallback defaults
        if api_base is None:
            api_base = (api_config.get("base_url") or "").strip()
            if not api_base:
                if self.platform == 'openai':
                    api_base = 'https://api.openai.com/v1'
                else:
                    server_host = cfg.get("server", {}).get("host", "0.0.0.0")
                    server_port = cfg.get("server", {}).get("port", 8000)
                    api_base = f"http://{server_host}:{server_port}/v1"

        if api_key is None:
            api_key = api_config.get("key", "EMPTY" if self.platform == 'vllm' else "")

        if model_name is None:
            if self.platform == 'openai':
                model_name = api_config.get('llm_model', 'gpt-4o')
            else:
                model_name = (
                    cfg.get("model", {})
                    .get("llm", {})
                    .get("name", "Qwen/Qwen3-30B-A3B-Instruct-2507")
                )

        if timeout is None:
            timeout = int(cfg.get("api", {}).get("timeout", 3600))

        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        self.model_name = model_name

    def load_materials_from_db(self, db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load materials metadata from SQLite database.

        Args:
            db_path: Path to the SQLite database file (defaults from config.yml)

        Returns:
            Dictionary containing file-level and page-level information
        """
        if db_path is None:
            db_path = self.cfg.get("paths", {}).get(
                "materials_db_path", "materials/processed_materials.db"
            )

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            raise FileNotFoundError(f"Database file not found at {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Read all materials from database
        cursor.execute("""
            SELECT id, original_filename, current_filename, status, 
                   description, needed, key_concept 
            FROM materials
        """)

        rows = cursor.fetchall()
        conn.close()

        # Organize data by file
        files_data = {}  # Map of original_filename -> {file_info, pages}

        for row in rows:
            (
                row_id,
                original_filename,
                current_filename,
                status,
                description,
                needed,
                key_concept,
            ) = row

            if original_filename not in files_data:
                files_data[original_filename] = {"file_info": None, "pages": []}

            # Check if this is a file-level summary (current_filename == 'all')
            if current_filename == "all":
                files_data[original_filename]["file_info"] = {
                    "id": row_id,
                    "filename": original_filename,
                    "description": description or "",
                    "key_concepts": json.loads(key_concept) if key_concept else [],
                }
            else:
                # This is a page-level entry
                # Extract page number from current_filename (e.g., "file_page_1.png" -> 1)
                try:
                    page_num = self._extract_page_number(current_filename)
                except Exception:
                    page_num = len(files_data[original_filename]["pages"]) + 1

                files_data[original_filename]["pages"].append(
                    {
                        "id": row_id,
                        "page_number": page_num,
                        "filename": current_filename,
                        "needed": needed == 1 if needed is not None else None,
                        "description": description or "",
                        "key_concept": key_concept or "",
                    }
                )

        # Sort pages by page number for each file
        for filename in files_data:
            files_data[filename]["pages"].sort(key=lambda x: x["page_number"])

        # Filter out files without valid file_info or without any needed pages
        valid_files_data = {}
        for filename, data in files_data.items():
            if data["file_info"] and data["file_info"]["description"]:
                # Check if there are any needed pages
                needed_pages = [p for p in data["pages"] if p.get("needed")]
                if needed_pages:
                    valid_files_data[filename] = data

        return valid_files_data

    def _prepare_structured_output(self, schema: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Prepare structured output parameters based on platform.
        
        Args:
            schema: JSON schema dictionary
            
        Returns:
            Tuple of (extra_body, response_format) where one will be None based on platform
        """
        if self.platform == 'openai':
            # OpenAI format: use response_format parameter
            schema_name = None
            if isinstance(schema, dict):
                schema_name = schema.get("title")
            if not schema_name:
                schema_name = "structured_output"
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True
                }
            }
            return None, response_format
        else:
            # vLLM format: use guided_json in extra_body
            extra_body = {"guided_json": schema}
            return extra_body, None

    def _extract_page_number(self, filename: str) -> int:
        """
        Extract page number from filename.
        Expected format: originalname_page_N.ext
        """
        import re

        match = re.search(r"_page_(\d+)", filename)
        if match:
            return int(match.group(1))
        # Fallback: try to find any number in the filename
        match = re.search(r"(\d+)", filename)
        if match:
            return int(match.group(1))
        return 0

    def generate_completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a completion using the LLM with structured output.

        Args:
            prompt: Text prompt to send to the model
            temperature: Sampling temperature (defaults from config.yml)
            top_p: Nucleus sampling parameter (defaults from config.yml)
            top_k: Top-k sampling parameter (defaults from config.yml)
            presence_penalty: Presence penalty (defaults from config.yml)
            max_tokens: Maximum number of tokens to generate (defaults from config.yml)
            guided_json: JSON schema for structured output
            verbose: If True, print timing information

        Returns:
            Dictionary with 'content' (parsed JSON) and 'metrics'
        """
        # Use platform-specific config values if not provided
        # For OpenAI, don't use any generation configs - only use explicitly provided parameters
        if self.platform == 'openai':
            # For OpenAI, keep None values as None - don't use any config defaults
            top_p_val = None
            top_k_val = None
        else:
            # For vLLM, use config values
            gen_config = self.cfg.get("generation", {}).get(self.platform, {}).get("llm", {})
            if temperature is None:
                temperature = float(gen_config.get("temperature", 0.3))
            if presence_penalty is None:
                presence_penalty = float(gen_config.get("presence_penalty", 0.0))
            if max_tokens is None:
                max_tokens = int(gen_config.get("max_tokens", 1024))
            
            # Only include top_p and top_k for vLLM platform
            top_p_val = None
            top_k_val = None
            if top_p is None:
                top_p_val = float(gen_config.get("top_p", 0.9))
            else:
                top_p_val = top_p
            if top_k is None:
                top_k_val = int(gen_config.get("top_k", 10))
            else:
                top_k_val = top_k

        messages = [{"role": "user", "content": prompt}]

        try:
            start_time = time.time()
            extra_body = {}
            response_format = None
            
            # Only add top_k to extra_body for vLLM
            if self.platform == 'vllm' and top_k_val is not None:
                extra_body["top_k"] = top_k_val
            
            if guided_json is not None:
                struct_extra_body, struct_response_format = self._prepare_structured_output(guided_json)
                if struct_extra_body:
                    extra_body.update(struct_extra_body)
                if struct_response_format:
                    response_format = struct_response_format

            create_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            
            # Only include generation parameters for vLLM or if explicitly provided for OpenAI
            if self.platform == 'vllm':
                create_kwargs["temperature"] = temperature
                create_kwargs["presence_penalty"] = presence_penalty
                create_kwargs["max_tokens"] = max_tokens
                if top_p_val is not None:
                    create_kwargs["top_p"] = top_p_val
            else:
                # For OpenAI, only include if not None/default
                if temperature is not None:
                    create_kwargs["temperature"] = temperature
                if presence_penalty is not None:
                    create_kwargs["presence_penalty"] = presence_penalty
                if max_tokens is not None:
                    create_kwargs["max_tokens"] = max_tokens
            
            # Only include extra_body if it has content
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            
            if response_format is not None:
                create_kwargs["response_format"] = response_format

            if self.platform == 'openai' and self.service_tier is not None:
                create_kwargs["service_tier"] = self.service_tier

            stream = self.client.chat.completions.create(**create_kwargs)

            content = ""
            first_token_time = None
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            for chunk in stream:
                # Track time to first token
                if (
                    first_token_time is None
                    and chunk.choices
                    and len(chunk.choices) > 0
                ):
                    if chunk.choices[0].delta.content:
                        first_token_time = time.time()

                # Collect content
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content

                # Extract usage information
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

            end_time = time.time()
            total_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else 0
            generation_time = (
                end_time - first_token_time if first_token_time else total_time
            )

            # Calculate metrics
            metrics = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "total_time": total_time,
                "ttft": ttft,
                "generation_time": generation_time,
                "pp_per_sec": prompt_tokens / ttft if ttft > 0 else 0,
                "tg_per_sec": completion_tokens / generation_time
                if generation_time > 0
                else 0,
            }

            if verbose:
                print(
                    f"  ✓ PP: {metrics['prompt_tokens']}, TG: {metrics['completion_tokens']}"
                )
                print(
                    f"  ✓ TTFT: {metrics['ttft']:.3f}s, TG/sec: {metrics['tg_per_sec']:.2f}, PP/sec: {metrics['pp_per_sec']:.2f}"
                )

            # Parse JSON response
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON response: {e}\nContent: {content}"
                )

            return {
                "content": parsed_content,
                "raw_content": content,
                "metrics": metrics,
            }

        except Exception as e:
            print(f"Error generating completion: {e}")
            raise

    def step1_select_file(
        self,
        question: str,
        wrong_answer: str,
        correct_answer: Optional[str] = None,
        materials_data: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Step 1: Select the most relevant file based on the question and wrong answer.

        Args:
            question: The multiple choice question text
            wrong_answer: The incorrect answer the student selected
            correct_answer: The correct answer (optional, for context)
            materials_data: Pre-loaded materials data (if None, will load from DB)
            db_path: Path to database (used if materials_data is None)
            verbose: If True, print progress information

        Returns:
            Dictionary with selected file information and reasoning
        """
        if verbose:
            print("=" * 80)
            print("STEP 1: File Selection")
            print("=" * 80)

        # Load materials if not provided
        if materials_data is None:
            if verbose:
                print("Loading materials from database...")
            materials_data = self.load_materials_from_db(db_path)

        if not materials_data:
            raise ValueError("No valid materials found in database")

        if verbose:
            print(f"Found {len(materials_data)} files with descriptions")

        # Build the prompt with file information
        file_selection_prompt_template = (
            self.cfg.get("prompts", {}).get("llm", {}).get("file_selection", "")
        )

        # Format materials information
        materials_info = []
        for filename, data in materials_data.items():
            file_info = data["file_info"]
            key_concepts_str = (
                ", ".join(file_info["key_concepts"][:5])
                if file_info["key_concepts"]
                else "N/A"
            )
            materials_info.append(
                f"File: {filename}\n"
                f"Key Concepts: {key_concepts_str}\n"
                f"Description: {file_info['description']}"
            )

        materials_list = "\n\n".join(materials_info)

        # Build full prompt
        full_prompt = f"""{file_selection_prompt_template}

Question: {question}

Student's Wrong Answer: {wrong_answer}
"""

        if correct_answer:
            full_prompt += f"Correct Answer: {correct_answer}\n"

        full_prompt += f"""
Available Materials:
{materials_list}

Analyze the question and the student's misconception, then select the most relevant file that will help them understand the correct concept."""

        # Define JSON schema for file selection
        file_selection_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_file": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["selected_file", "reasoning"],
        }
        if verbose:
            print("Sending request to select file...")

        result = self.generate_completion(
            full_prompt, guided_json=file_selection_schema, verbose=verbose
        )

        selected_file = result["content"]["selected_file"]
        reasoning = result["content"]["reasoning"]

        if verbose:
            print(f"\n✓ Selected File: {selected_file}")
            print(f"  Reasoning: {reasoning}")

        # Validate that the selected file exists in our data
        if selected_file not in materials_data:
            # Try to find a close match
            for filename in materials_data.keys():
                if selected_file in filename or filename in selected_file:
                    selected_file = filename
                    break
            else:
                raise ValueError(
                    f"Selected file '{selected_file}' not found in available materials"
                )

        return {
            "selected_file": selected_file,
            "reasoning": reasoning,
            "file_data": materials_data[selected_file],
            "metrics": result["metrics"],
        }

    def step2_select_pages(
        self,
        question: str,
        wrong_answer: str,
        selected_file: str,
        file_data: Dict[str, Any],
        correct_answer: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Step 2: Select the most relevant page range within the selected file.

        Args:
            question: The multiple choice question text
            wrong_answer: The incorrect answer the student selected
            selected_file: The filename selected in step 1
            file_data: The file's page data from materials_data
            correct_answer: The correct answer (optional, for context)
            verbose: If True, print progress information

        Returns:
            Dictionary with selected page range and reasoning
        """
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 2: Page Range Selection")
            print("=" * 80)

        # Filter to only needed pages
        needed_pages = [p for p in file_data["pages"] if p.get("needed")]

        if not needed_pages:
            raise ValueError(f"No relevant pages found in {selected_file}")

        if verbose:
            print(f"Analyzing {len(needed_pages)} relevant pages from {selected_file}")

        # Build the prompt with page information
        page_selection_prompt_template = (
            self.cfg.get("prompts", {}).get("llm", {}).get("page_selection", "")
        )

        # Format page information
        pages_info = []
        for page in needed_pages:
            pages_info.append(
                f"Page {page['page_number']}:\n"
                f"  Key Concept: {page['key_concept']}\n"
                f"  Description: {page['description']}"
            )

        pages_list = "\n\n".join(pages_info)

        # Build full prompt
        full_prompt = f"""{page_selection_prompt_template}

Question: {question}

Student's Wrong Answer: {wrong_answer}
"""

        if correct_answer:
            full_prompt += f"Correct Answer: {correct_answer}\n"

        full_prompt += f"""
Selected Material: {selected_file}

Available Pages:
{pages_list}

Analyze the question and the student's misconception, then select a focused range of consecutive pages (3-5 pages maximum) that will help them understand the correct concept. Use the actual page numbers from the list above."""

        # Define JSON schema for page selection
        page_selection_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "start_page": {"type": "integer"},
                "end_page": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": ["start_page", "end_page", "reasoning"],
        }
        if verbose:
            print("\nGenerating page range recommendation...")

        result = self.generate_completion(
            full_prompt, guided_json=page_selection_schema, verbose=verbose
        )

        start_page = result["content"]["start_page"]
        end_page = result["content"]["end_page"]
        reasoning = result["content"]["reasoning"]

        # Validate page range
        page_numbers = [p["page_number"] for p in needed_pages]
        min_page = min(page_numbers)
        max_page = max(page_numbers)

        # Adjust if out of bounds
        if start_page < min_page:
            start_page = min_page
        if end_page > max_page:
            end_page = max_page
        if start_page > end_page:
            start_page, end_page = end_page, start_page

        # Get the actual pages in the range
        selected_pages = [
            p for p in needed_pages if start_page <= p["page_number"] <= end_page
        ]

        if verbose:
            print(
                f"\n✓ Selected Page Range: {start_page} - {end_page} ({len(selected_pages)} pages)"
            )
            print(f"  Reasoning: {reasoning}")

        return {
            "start_page": start_page,
            "end_page": end_page,
            "num_pages": len(selected_pages),
            "pages": selected_pages,
            "reasoning": reasoning,
            "metrics": result["metrics"],
        }

    def recommend(
        self,
        question: str,
        wrong_answer: str,
        correct_answer: Optional[str] = None,
        db_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete two-step recommendation: select file and then page range.

        Args:
            question: The multiple choice question text
            wrong_answer: The incorrect answer the student selected
            correct_answer: The correct answer (optional, for context)
            db_path: Path to the SQLite database file (defaults from config.yml)
            verbose: If True, print progress information

        Returns:
            Dictionary with complete recommendation results
        """
        if verbose:
            print("\n" + "=" * 80)
            print("EDUCATIONAL MATERIAL RECOMMENDATION SYSTEM")
            print("=" * 80)
            print(f"\nQuestion: {question}")
            print(f"Wrong Answer: {wrong_answer}")
            if correct_answer:
                print(f"Correct Answer: {correct_answer}")

        # Load materials
        materials_data = self.load_materials_from_db(db_path)

        # Step 1: Select file
        step1_result = self.step1_select_file(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            materials_data=materials_data,
            verbose=verbose,
        )

        # Step 2: Select pages
        step2_result = self.step2_select_pages(
            question=question,
            wrong_answer=wrong_answer,
            selected_file=step1_result["selected_file"],
            file_data=step1_result["file_data"],
            correct_answer=correct_answer,
            verbose=verbose,
        )

        # Combine results
        recommendation = {
            "question": question,
            "wrong_answer": wrong_answer,
            "correct_answer": correct_answer,
            "step1": {
                "selected_file": step1_result["selected_file"],
                "reasoning": step1_result["reasoning"],
                "metrics": step1_result["metrics"],
            },
            "step2": {
                "start_page": step2_result["start_page"],
                "end_page": step2_result["end_page"],
                "num_pages": step2_result["num_pages"],
                "reasoning": step2_result["reasoning"],
                "metrics": step2_result["metrics"],
            },
        }

        if verbose:
            print("\n" + "=" * 80)
            print("RECOMMENDATION SUMMARY")
            print("=" * 80)
            print(f"Recommended Material: {recommendation['step1']['selected_file']}")
            print(
                f"Recommended Pages: {recommendation['step2']['start_page']} - {recommendation['step2']['end_page']}"
            )
            print(f"\nFile Selection Reasoning: {recommendation['step1']['reasoning']}")
            print(f"Page Selection Reasoning: {recommendation['step2']['reasoning']}")
            print("\n" + "=" * 80)
            print("PERFORMANCE METRICS")
            print("=" * 80)
            step1_metrics = recommendation["step1"]["metrics"]
            step2_metrics = recommendation["step2"]["metrics"]
            total_time = step1_metrics["total_time"] + step2_metrics["total_time"]
            total_tokens = step1_metrics["total_tokens"] + step2_metrics["total_tokens"]
            print(f"Total Time: {total_time:.2f}s")
            print(f"Total Tokens: {total_tokens}")
            print(
                f"Step 1 - File Selection: {step1_metrics['total_time']:.2f}s, {step1_metrics['total_tokens']} tokens"
            )
            print(
                f"Step 2 - Page Selection: {step2_metrics['total_time']:.2f}s, {step2_metrics['total_tokens']} tokens"
            )

        return recommendation
