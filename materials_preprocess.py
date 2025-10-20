#!/usr/bin/env python3
"""
Process PDF materials into individual page images and create a SQLite database manifest.
"""

import os
import sqlite3
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from config_loader import load_config

def process_pdfs():
    # Load configuration from YAML
    cfg = load_config()
    # Define directories from YAML config (with safe defaults)
    pdf_dir = Path(cfg.get('paths', {}).get('pdf_dir', "materials/pdf"))
    output_dir = Path(cfg.get('paths', {}).get('materials_dir', "materials/processed"))
    db_path = Path(cfg.get('paths', {}).get('materials_db_path', "materials/processed_materials.db"))
    
    # Clean up previous processed data if it exists
    if output_dir.exists():
        print(f"Removing previous processed folder: {output_dir}")
        shutil.rmtree(output_dir)
    
    if db_path.exists():
        print(f"Removing previous database file: {db_path}")
        db_path.unlink()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    # Rename PDF files to remove spaces
    renamed_pdf_files = []
    for pdf_path in pdf_files:
        if ' ' in pdf_path.name:
            new_name = pdf_path.name.replace(' ', '')
            new_path = pdf_path.parent / new_name
            print(f"Renaming: {pdf_path.name} -> {new_name}")
            pdf_path.rename(new_path)
            renamed_pdf_files.append(new_path)
        else:
            renamed_pdf_files.append(pdf_path)
    
    pdf_files = sorted(renamed_pdf_files)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Create database and table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with the same four columns as the original CSV structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            current_filename TEXT NOT NULL,
            status TEXT,
            description TEXT,
            needed INTEGER,
            key_concept TEXT
        )
    ''')
    conn.commit()
    
    # Process each PDF
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        
        # Get base filename without extension and remove spaces
        base_name = pdf_path.stem.replace(' ', '')
        
        try:
            # Convert PDF to images
            images = convert_from_path(str(pdf_path))
            
            print(f"  Converting {len(images)} pages...")
            
            # Save each page as JPG
            for page_num, image in enumerate(images, start=1):
                # Create output filename
                output_filename = f"{base_name}_page_{page_num}.jpg"
                output_path = output_dir / output_filename
                
                # Save image
                image.save(str(output_path), "JPEG", quality=95)
                
                # Insert into database
                cursor.execute('''
                    INSERT INTO materials (original_filename, current_filename, status, description, needed, key_concept)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (pdf_path.name, output_filename, '', '', None, ''))
                
                print(f"  ✓ Saved: {output_filename}")
        
        except Exception as e:
            print(f"  ✗ Error processing {pdf_path.name}: {e}")
        
        # Add summary row for this file
        cursor.execute('''
            INSERT INTO materials (original_filename, current_filename, status, description, needed, key_concept)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pdf_path.name, 'all', '', '', None, ''))
    
    # Commit changes and close database
    conn.commit()
    
    # Get count of inserted records
    cursor.execute('SELECT COUNT(*) FROM materials')
    record_count = cursor.fetchone()[0]
    
    conn.close()
    
    if record_count > 0:
        print(f"\n✓ Database created: {db_path}")
        print(f"✓ Total records created: {record_count}")
        print(f"✓ Images saved to: {output_dir}")
    else:
        print("\n✗ No records were created")

if __name__ == "__main__":
    process_pdfs()

