#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a PDF comparison document showing images and descriptions from three different models.
"""

import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html import escape
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Spacer, Paragraph, Image
from reportlab.lib import colors
from PIL import Image as PILImage


def load_descriptions_from_db(db_path: str) -> Dict[Tuple[str, str], Dict]:
    """
    Load descriptions from a database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dictionary mapping (original_filename, current_filename) -> description data
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT original_filename, current_filename, description, key_concept, status, needed
        FROM materials
        WHERE current_filename != 'all'
    """)
    
    results = {}
    for row in cursor.fetchall():
        key = (row['original_filename'], row['current_filename'])
        # Convert needed field: 1 -> "true", 0 -> "false", None -> "false"
        needed_val = row['needed']
        if needed_val is None:
            needed_str = "false"
        elif needed_val == 1:
            needed_str = "true"
        else:
            needed_str = "false"
        
        results[key] = {
            'description': row['description'] or '',
            'key_concept': row['key_concept'] or '',
            'status': row['status'] or '',
            'needed': needed_str
        }
    
    conn.close()
    return results


def get_all_pages(db_paths: List[str]) -> List[Tuple[str, str, int]]:
    """
    Get a list of all unique pages (original_filename, current_filename, id) from all databases.
    
    Args:
        db_paths: List of database paths
        
    Returns:
        List of (original_filename, current_filename, id) tuples, sorted by ID
    """
    # Use a dict to store pages with their minimum ID across all databases
    pages_dict = {}
    
    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, original_filename, current_filename
            FROM materials
            WHERE current_filename != 'all'
            ORDER BY id
        """)
        for row in cursor.fetchall():
            page_id = row[0]
            original_filename = row[1]
            current_filename = row[2]
            page_key = (original_filename, current_filename)
            
            # Keep the minimum ID seen across all databases for this page
            if page_key not in pages_dict or page_id < pages_dict[page_key]:
                pages_dict[page_key] = page_id
        conn.close()
    
    # Convert to list of tuples and sort by ID
    all_pages = [(original, current, page_id) 
                 for (original, current), page_id in pages_dict.items()]
    all_pages.sort(key=lambda x: x[2])  # Sort by ID
    
    return all_pages


def wrap_text(text: str, max_width: int) -> str:
    """
    Simple text wrapping for long lines.
    """
    if not text:
        return ""
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_width and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


def generate_comparison_pdf(
    output_path: str,
    db_paths: Dict[str, str],
    materials_dir: str,
    model_names: Dict[str, str]
):
    """
    Generate a PDF comparison document.
    
    Args:
        output_path: Path to save the output PDF
        db_paths: Dictionary mapping model key to database path
        materials_dir: Directory containing the JPG images
        model_names: Dictionary mapping model key to display name
    """
    materials_dir_path = Path(materials_dir)
    
    # Load descriptions from all databases
    print("Loading descriptions from databases...")
    descriptions = {}
    for model_key, db_path in db_paths.items():
        print(f"  Loading {model_key}...")
        descriptions[model_key] = load_descriptions_from_db(db_path)
    
    # Get all unique pages
    print("Collecting all pages...")
    all_pages = get_all_pages(list(db_paths.values()))
    print(f"Found {len(all_pages)} unique pages")
    
    # Create PDF document with portrait orientation
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Build story (content)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#000000'),
        spaceAfter=4,
        alignment=1,  # Center alignment
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        leading=13,
        fontName='Helvetica'
    )
    
    model_title_style = ParagraphStyle(
        'ModelTitle',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#0066cc'),
        spaceAfter=4,
        spaceBefore=6,
        fontName='Helvetica-Bold'
    )
    
    # Process each page
    for page_idx, (original_filename, current_filename, page_id) in enumerate(all_pages, 1):
        print(f"Processing page {page_idx}/{len(all_pages)} (ID: {page_id}): {current_filename}")
        
        # Find image path
        image_path = materials_dir_path / current_filename
        
        if not image_path.exists():
            print(f"  Warning: Image not found: {image_path}")
            continue
        
        # Get image dimensions
        try:
            pil_img = PILImage.open(image_path)
            img_width, img_height = pil_img.size
            
            # Calculate scaling to fit available width in portrait mode
            # Portrait: 8.5" width - 1" margins = 7.5" available width
            # Portrait: 11" height - 1" margins = 10" available height
            # Reserve space for image at top and text below
            max_height = 3.0 * inch  # Limit height for image (leave ~7" for text)
            max_width = 7.0 * inch   # Nearly full width available
            
            scale_w = max_width / img_width
            scale_h = max_height / img_height
            scale = min(scale_w, scale_h, 1.0)  # Don't scale up
            
            scaled_width = img_width * scale
            scaled_height = img_height * scale
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue
        
        # Create table for this page
        # Left column: Image
        # Right column: Three model descriptions
        
        # Prepare descriptions - dynamically iterate through all models
        desc_data = []
        for model_key in db_paths.keys():
            if model_key in descriptions:
                page_data = descriptions[model_key].get((original_filename, current_filename), {})
                desc_text = page_data.get('description', 'No description available')
                key_concept = page_data.get('key_concept', '')
                needed = page_data.get('needed', 'false')
                
                # Escape HTML special characters
                desc_text_escaped = escape(str(desc_text))
                key_concept_escaped = escape(str(key_concept))
                needed_escaped = escape(str(needed))
                
                # Build text with needed, key concept, and description
                parts = [f"<b>Needed:</b> {needed_escaped}"]
                
                if key_concept:
                    parts.append(f"<b>Key Concept:</b> {key_concept_escaped}")
                
                parts.append(desc_text_escaped)
                
                full_text = "<br/>".join(parts)  # Use single <br/> for tighter spacing
                desc_data.append((model_key, full_text))
            else:
                desc_data.append((model_key, 'Description not available'))
        
        # Create table data with new layout:
        # Row 1: Header with filename and page number
        # Row 2: Image (spanning full width)
        # Row 3+: One row per model (dynamic based on number of models)
        
        table_data = []
        
        # Extract page number from filename (e.g., "Day13-Phys1111_09_15_2025_Forces_page_1.jpg")
        page_match = re.search(r'_page_(\d+)\.jpg$', current_filename)
        page_number = page_match.group(1) if page_match else "?"
        
        # Create header text with filename and page number
        header_text = f"<b>{original_filename}</b> - Page {page_number}"
        header_cell = Paragraph(header_text, header_style)
        table_data.append([header_cell])
        
        # Second row: Image centered, spanning all columns
        img_cell = Image(str(image_path), width=scaled_width, height=scaled_height)
        table_data.append([img_cell])
        
        # Add one row per model (dynamically based on desc_data)
        for model_key, desc_text in desc_data:
            model_display_name = model_names.get(model_key, model_key)
            model_text = f"<b>{model_display_name}</b><br/>{desc_text}"
            model_content = Paragraph(model_text, body_style)
            table_data.append([model_content])
        
        # Create table - portrait gives us ~7.5" width (8.5" - 1" margins)
        # Single column spanning full width
        table = Table(table_data, colWidths=[7.5*inch])
        
        # Style the table dynamically based on number of rows
        num_models = len(desc_data)
        last_row = num_models + 1  # 0 is header, 1 is image, 2 to num_models+1 are model descriptions
        
        style_commands = [
            # Row 0: Header - center and style with light purple background
            ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#e6d5f5')),  # Light purple
            ('BOX', (0, 0), (0, 0), 2, colors.black),
            
            # Row 1: Image - center it with border
            ('BOX', (0, 1), (0, 1), 2, colors.black),
            ('VALIGN', (0, 1), (0, 1), 'MIDDLE'),          # Center image vertically
            ('ALIGN', (0, 1), (0, 1), 'CENTER'),           # Center image horizontally
            
            # All model rows: Text alignment
            ('VALIGN', (0, 2), (0, last_row), 'TOP'),      # Align text to top
            ('ALIGN', (0, 2), (0, last_row), 'LEFT'),      # Left align text
            
            # Padding - reduce slightly to fit more content vertically
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]
        
        # Add borders around each model section (starting from row 2)
        for i in range(2, num_models + 2):
            style_commands.append(('BOX', (0, i), (0, i), 2, colors.black))
        
        # Add alternating background colors for model sections (starting from row 2)
        # Alternate between light green and light blue
        for i in range(2, num_models + 2):
            if (i - 2) % 2 == 0:  # Adjust index for alternation
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor('#d4f4dd')))  # Light green
            else:
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor('#d4e9f7')))  # Light blue
        
        table.setStyle(TableStyle(style_commands))
        
        story.append(table)
        story.append(PageBreak())  # Remove extra spacer to save space
    
    # Build PDF
    print(f"\nGenerating PDF: {output_path}")
    doc.build(story)
    print(f"Done! PDF saved to: {output_path}")


def main():
    """Main function."""
    # Database paths
    db_paths = {
        'gpt5': 'results/DBs/gpt5-processed_materials.db',
        '235B': 'results/DBs/235B-processed_materials.db',
        '30B': 'results/DBs/30B-processed_materials.db'
        # 'empty': 'materials/processed_materials.db'
    }
    
    # Model display names
    model_names = {
        'gpt5': 'GPT-5',
        '235B': 'Qwen3-VL-235B-A22B-Instruct',
        '30B': 'Qwen3-VL-30B-A3B-Instruct'
        # 'empty': 'Empty'
    }
    
    # Materials directory
    materials_dir = 'materials/processed'
    
    # Output PDF path
    output_path = 'results/comparison.pdf'
    
    # Verify databases exist
    for model_key, db_path in db_paths.items():
        if not Path(db_path).exists():
            print(f"Error: Database not found: {db_path}")
            return
    
    # Generate PDF
    generate_comparison_pdf(
        output_path=output_path,
        db_paths=db_paths,
        materials_dir=materials_dir,
        model_names=model_names
    )


if __name__ == '__main__':
    main()

