#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default prompt templates for description generation.
"""

# Default prompt for single page analysis
SINGLE_PAGE_PROMPT_DEFAULT = (
    "Return only a JSON object with fields exactly: needed (boolean), key_concept (string), description (string). "
    "Rules: needed=true only if the page teaches a substantive concept, method, worked example, or definition. "
    "Mark needed=false for pages that are: title/cover, course schedule/timeline, syllabus outline, table of contents, announcements, logistics (deadlines, office hours, emails), exam/quiz information, instructions to reflect/prepare, grading/policies, decorative/quote/blank pages, single-panel comics or cartoons, inspirational quotes, and high-level overview/agenda slides that summarize sections without teaching details. "
    "key_concept is a 2-6 word phrase naming the primary concept taught on this page; if multiple topics appear, choose the most central. "
    "description is 1-2 short sentences defining the concept and what a student needs to know to answer multiple-choice questions about it; keep under 40 words; plain text; no lists; ignore headers/footers and long OCR passages. Output JSON only with no extra text."
)

# Default prompt for batch analysis
BATCH_PROMPT_DEFAULT = (
    "Return only a JSON object with fields exactly: key_concept (array of 5 strings), description (string). Do not include a needed field. "
    "The selected pages will already exclude administrative content (title, schedule, announcements, logistics, exam info, syllabus/TOC). "
    "Rules: key_concept must contain exactly five phrases (each 2-6 words) capturing the main concepts covered across the selected pages. "
    "description is 3-4 sentences summarizing the overall topic, key ideas, and typical problem types or skills assessed; concise, <=100 words; plain text; no bullets; avoid page-level details. Output JSON only with no extra text."
)


def get_prompts_from_config(cfg: dict) -> tuple[str, str]:
    """
    Get prompt templates from config, falling back to defaults.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (single_prompt, batch_prompt)
    """
    prompt_cfg = cfg.get('prompts', {}).get('vlm', {})
    single_prompt = prompt_cfg.get('single') or SINGLE_PAGE_PROMPT_DEFAULT
    batch_prompt = prompt_cfg.get('batch') or BATCH_PROMPT_DEFAULT
    return single_prompt, batch_prompt

