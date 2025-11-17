#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API client with retry logic for OpenAI-compatible endpoints.
"""

import time
import base64
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from openai_cost_calculator import estimate_cost_typed

from .types import PerformanceMetrics, GenerationResult


class APIClient:
    """Handles API communication with retry logic."""
    
    # Retry configuration
    RETRY_DELAYS = [2, 5, 10, 15, 30]  # Exponential backoff delays
    MAX_DELAY = 30  # Maximum delay cap
    
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        service_tier: Optional[str] = None
    ):
        """
        Initialize API client.
        
        Args:
            client: OpenAI client instance
            model_name: Name of the model to use
            service_tier: Optional service tier for API requests
        """
        self.client = client
        self.model_name = model_name
        self.service_tier = service_tier
    
    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def create_message_with_image(
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
            image_url = image_path
        else:
            base64_image = APIClient.encode_image_to_base64(image_path)
            image_url = f"data:image/jpeg;base64,{base64_image}"
        
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    
    @staticmethod
    def create_message_with_multiple_images(
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
        
        for image_path in image_paths:
            if use_url:
                image_url = image_path
            else:
                base64_image = APIClient.encode_image_to_base64(image_path)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content}]
    
    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Optional[Dict[str, float]]:
        """
        Calculate cost using openai-cost-calculator.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Dict with input_cost, output_cost, total_cost or None if calculation fails
        """
        if prompt_tokens == 0 and completion_tokens == 0:
            return None
        
        # Create mock objects for cost calculation
        class MockUsage:
            def __init__(self, prompt_tokens: int, completion_tokens: int):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens
                self.total_tokens = prompt_tokens + completion_tokens
        
        class MockResponse:
            def __init__(self, model: str, usage: MockUsage):
                self.model = model
                self.usage = usage
        
        try:
            mock_usage = MockUsage(prompt_tokens, completion_tokens)
            mock_response = MockResponse(self.model_name, mock_usage)
            cost_details = estimate_cost_typed(mock_response)
            
            return {
                'input_cost': float(cost_details.prompt_cost_uncached + cost_details.prompt_cost_cached),
                'output_cost': float(cost_details.completion_cost),
                'total_cost': float(cost_details.total_cost)
            }
        except Exception as e:
            print(f"Warning: Could not calculate cost: {e}")
            return None
    
    def _stream_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> tuple[str, PerformanceMetrics]:
        """
        Stream completion from API and collect metrics.
        
        Args:
            messages: List of message dictionaries
            response_format: Optional structured output format
            temperature: Sampling temperature
            presence_penalty: Presence penalty
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (content, metrics)
        """
        start_time = time.time()
        
        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        # Add optional parameters
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if presence_penalty is not None:
            create_kwargs["presence_penalty"] = presence_penalty
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        if self.service_tier is not None:
            create_kwargs["service_tier"] = self.service_tier
        
        stream = self.client.chat.completions.create(**create_kwargs)
        
        content = ""
        first_token_time: Optional[float] = None
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
        ttft = (first_token_time - start_time) if first_token_time else 0.0
        generation_time = (end_time - first_token_time) if first_token_time else total_time
        
        # Calculate cost
        cost_details = self._calculate_cost(prompt_tokens, completion_tokens)
        
        # Build metrics
        metrics = PerformanceMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_time=total_time,
            ttft=ttft,
            generation_time=generation_time,
            pp_per_sec=prompt_tokens / ttft if ttft > 0 else 0.0,
            tg_per_sec=completion_tokens / generation_time if generation_time > 0 else 0.0,
            input_cost=cost_details['input_cost'] if cost_details else None,
            output_cost=cost_details['output_cost'] if cost_details else None,
            total_cost=cost_details['total_cost'] if cost_details else None
        )
        
        return content, metrics
    
    def generate_with_retry(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False
    ) -> tuple[str, PerformanceMetrics]:
        """
        Generate completion with automatic retry on retriable errors.
        
        Args:
            messages: List of message dictionaries
            response_format: Optional structured output format
            temperature: Sampling temperature
            presence_penalty: Presence penalty
            max_tokens: Maximum tokens to generate
            verbose: If True, print retry information
            
        Returns:
            Tuple of (content, metrics)
            
        Raises:
            APIError: For non-retriable errors
            Exception: For other errors
        """
        attempt = 0
        
        while True:
            try:
                return self._stream_completion(
                    messages,
                    response_format,
                    temperature,
                    presence_penalty,
                    max_tokens
                )
            
            except (APIError, APIConnectionError, APITimeoutError) as e:
                # Check if it's a retriable error
                is_retriable = False
                if isinstance(e, APIError):
                    status_code = getattr(e, 'status_code', None)
                    if status_code in [500, 502, 503, 504]:
                        is_retriable = True
                elif isinstance(e, (APIConnectionError, APITimeoutError)):
                    is_retriable = True
                
                if is_retriable:
                    # Calculate delay with exponential backoff
                    if attempt < len(self.RETRY_DELAYS):
                        delay = self.RETRY_DELAYS[attempt]
                    else:
                        delay = self.MAX_DELAY
                    
                    attempt += 1
                    if verbose:
                        print(f"  ⚠ Retriable error (attempt {attempt}): {e}")
                        print(f"  ⏳ Retrying in {delay} seconds...")
                    else:
                        print(f"  ⚠ Retriable error (attempt {attempt}): {e}, retrying in {delay}s...")
                    
                    time.sleep(delay)
                    continue
                else:
                    # Not retriable - fail immediately
                    print(f"Error generating description: {e}")
                    raise
            
            except Exception as e:
                # Non-retriable errors should not be retried
                print(f"Error generating description: {e}")
                raise

