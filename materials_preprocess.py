#!/usr/bin/env python3
"""
Process PDF materials into individual page images and create a CSV manifest.
"""

import os
import csv
import shutil
from pathlib import Path
from pdf2image import convert_from_path

def process_pdfs():
    # Define directories
    pdf_dir = Path("materials/pdf")
    output_dir = Path("materials/processed")
    csv_path = Path("materials/processed_materials.csv")
    
    # Clean up previous processed data if it exists
    if output_dir.exists():
        print(f"Removing previous processed folder: {output_dir}")
        shutil.rmtree(output_dir)
    
    if csv_path.exists():
        print(f"Removing previous CSV file: {csv_path}")
        csv_path.unlink()
    
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
    
    # Prepare CSV data
    csv_data = []
    
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
                
                # Add to CSV data
                csv_data.append({
                    'original_filename': pdf_path.name,
                    'current_filename': output_filename,
                    'status': '',
                    'description': ''
                })
                
                print(f"  ✓ Saved: {output_filename}")
        
        except Exception as e:
            print(f"  ✗ Error processing {pdf_path.name}: {e}")
    
    # Write CSV file
    if csv_data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['original_filename', 'current_filename', 'status', 'description']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\n✓ CSV manifest created: {csv_path}")
        print(f"✓ Total images created: {len(csv_data)}")
        print(f"✓ Images saved to: {output_dir}")
    else:
        print("\n✗ No images were created")

if __name__ == "__main__":
    process_pdfs()

