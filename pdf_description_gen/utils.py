#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for PDF description generation.
"""

import re
from typing import Optional


def extract_page_number(filename: str) -> Optional[int]:
    """
    Extract page number from filename.
    Expected format: originalname_page_N.ext (e.g., "Day13-Phys1111_09_15_2025_Forces_page_1.jpg")
    
    Args:
        filename: The filename to extract page number from
        
    Returns:
        Page number if found, None otherwise
    """
    match = re.search(r"_page_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

