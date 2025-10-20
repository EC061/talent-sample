## Environment Setup

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Usage

```bash
python test_inference.py
```

## Materials Preprocessing

```bash
sudo apt install poppler-utils # for pdf2image
python materials_preprocess.py
```

## Materials Description Generation

The system now supports two processing modes:

### 1. Individual Page Processing
- Each page of a PDF is processed separately
- Row with `current_filename` = specific page (e.g., `Day13-Phys1111_09_15_2025_Forces_page_1.jpg`)
- Each page gets its own description and status

### 2. Batch Processing (All Images from Same File)
- When `current_filename` = `"all"`, the system:
  1. Collects all images from the same `original_filename`
  2. Sends all images together to the vision model in a single request
  3. Generates a combined description for the entire file
  4. Updates the "all" row with the batch description and status

### Database Structure
The `processed_materials.db` SQLite database contains a `materials` table with:
- `original_filename`: The original PDF filename
- `current_filename`: 
  - Page files: `{base_name}_page_{N}.jpg` (individual processing)
  - Batch row: `"all"` (batch processing of all pages)
- `status`: Processing status (`processed`, `error`, or empty)
- `description`: Generated description text

### Processing Example

```bash
python materials_description_gen.py
```

The script will:
1. Skip already processed rows (status = "processed")
2. Process individual page rows one by one
3. When encountering an "all" row:
   - Collect all page images from that file
   - Send them as a batch to the model
   - Generate a comprehensive description

### Advantages of Batch Processing
- More efficient API usage (fewer requests)
- Better context understanding (model sees all pages together)
- Comprehensive document-level analysis
- Reduced latency for multi-page documents

## Workflow Example

Given a PDF with 5 pages, the database will contain:

```
original_filename,current_filename,status,description
myfile.pdf,myfile_page_1.jpg,,
myfile.pdf,myfile_page_2.jpg,,
myfile.pdf,myfile_page_3.jpg,,
myfile.pdf,myfile_page_4.jpg,,
myfile.pdf,myfile_page_5.jpg,,
myfile.pdf,all,,
```

**Processing happens in two ways:**

1. **If you process individual pages** (skip "all" rows):
   - Page 1 gets analyzed individually → description A
   - Page 2 gets analyzed individually → description B
   - etc.

2. **If you process "all" rows** (skip page rows):
   - System collects pages 1-5
   - Sends all 5 images together to the model
   - Generates unified description → description for entire document

This allows flexible processing strategies depending on your needs!

## Technical Summary

The system consists of several key components:

1. **Data Pipeline**:
   - PDF files are converted to images using `pdf2image`
   - Images are stored in a `processed_images` directory
   - `processed_materials.db` SQLite database tracks the status and descriptions of each image

2. **Description Generation**:
   - The `materials_description_gen.py` script processes the database
   - It uses a vision model (`vision_model.py`) to generate descriptions
   - The model is designed to understand the structure and content of the document

3. **Model Architecture**:
   - The vision model is a pre-trained model (e.g., CLIP, OpenAI)
   - It is fine-tuned on a specific dataset of materials descriptions
   - The model's output is a text description of the image content

4. **API Integration**:
   - The system interacts with a vision API (e.g., OpenAI, Hugging Face)
   - It sends batches of images to the API for processing
   - The API returns combined descriptions for the entire file

5. **Error Handling**:
   - The script logs errors and updates the database with `error` status
   - It retries failed requests and maintains a retry mechanism

6. **Performance Optimization**:
   - Batch processing reduces the number of API calls
   - Efficient image loading and processing
   - Parallel processing of individual pages and batches

7. **Scalability**:
   - The system can handle large numbers of pages and files
   - It can be easily extended to support new models or data sources

8. **Flexibility**:
   - Users can choose to process individual pages or the entire file
   - The system adapts to different document structures and formats

9. **Maintainability**:
   - Clear code organization and modular design
   - Easy to debug and modify components
   - Comprehensive logging for monitoring and debugging

## Implementation Details

### Key Changes Made

#### 1. New Method: `create_message_with_multiple_images()`
This method in `MaterialsDescriptionGenerator` class handles sending multiple images to the vision API:

```python
def create_message_with_multiple_images(
    self,
    image_paths: List[str],
    prompt: str,
    use_url: bool = False
) -> List[Dict[str, Any]]:
```

- Takes a list of image paths (or URLs)
- Encodes each image as base64 if not using URLs
- Creates a single API message with all images and the prompt
- Returns the formatted message for the API call

#### 2. Enhanced `process_materials_db()` Method

The processing now happens in two passes:

**Pass 1: Identify Processing Tasks**
- Scans the SQLite database for rows to process
- Identifies "all" rows and maps them to their page indices
- Tracks which pages belong to which PDF

**Pass 2: Execute Processing Tasks**
- For each row marked for processing:
  - **Individual pages**: Sends single image to API, stores description
  - **Batch "all" rows**: Collects all pages, sends together to API, stores combined description

#### 3. Row Structure
```
original_filename: Immutable (original PDF name)
current_filename: "page_N.jpg" or "all"
status: "processed", "error", or empty
description: Generated text or error message
```

#### 4. Task Types
- `'individual'`: Process a single page independently
- `'batch'`: Process all pages from the same file together

### Processing Logic Flow

```
1. Load database with current_filename values
   ├── If "all" row exists for a file
   │   └── Collect all page indices for that file
   └── If page rows exist
       └── Add to processing queue

2. For each task in queue:
   ├── If type == 'individual':
   │   └── Send single image to API
   ├── If type == 'batch':
   │   ├── Collect all image paths for original_filename
   │   └── Send all images to API in single request

3. Update database with results
   ├── Set status = 'processed' or 'error'
   └── Set description = generated text or error message

4. Generate summary report with metrics
```

### Configuration
The system respects config.env settings:
- `API_BASE_URL`: Vision API endpoint
- `API_KEY`: Authentication key
- `MODEL_NAME`: Model to use
- `GENERATION_TEMPERATURE`: Sampling temperature
- `GENERATION_MAX_TOKENS`: Maximum response length
- `BATCH_SIZE`: Number of images per batch (optional)