#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point script for PDF description generation.
Maintains backwards compatibility with the original interface.
"""

from pdf_description_gen import MaterialsDescriptionGenerator


def main() -> None:
    """Main function to demonstrate usage."""
    # Initialize generator with OpenAI-compatible endpoint using YAML config
    generator = MaterialsDescriptionGenerator()
    
    # Process all materials from database (params loaded from YAML)
    generator.process_materials_db()


if __name__ == "__main__":
    main()

