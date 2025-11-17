#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON schemas for structured output.
"""

from typing import Dict, Any


def get_single_page_schema() -> Dict[str, Any]:
    """
    Get JSON schema for single page description.
    
    Returns:
        JSON schema dict with fields: needed, key_concept, description
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "needed": {"type": "boolean"},
            "key_concept": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["needed", "key_concept", "description"]
    }


def get_batch_schema() -> Dict[str, Any]:
    """
    Get JSON schema for batch description.
    
    Returns:
        JSON schema dict with fields: key_concept (array of 5 strings), description
    """
    return {
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


def prepare_structured_output(schema: Dict[str, Any]) -> tuple[None, Dict[str, Any]]:
    """
    Prepare structured output parameters for OpenAI API.
    
    Args:
        schema: JSON schema dictionary
        
    Returns:
        Tuple of (None, response_format dict)
    """
    schema_name = schema.get('title', 'structured_output')
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True
        }
    }
    return None, response_format

