# PDF Description Generation Package

A well-structured, type-hinted Python package for generating descriptions of educational materials using OpenAI-compatible vision-language models.

## Package Structure

```
pdf_description_gen/
├── __init__.py           # Package exports
├── generator.py          # Main generator class
├── api_client.py         # API client with retry logic
├── database.py           # Database operations
├── logger.py             # Markdown logging
├── schemas.py            # JSON schemas for structured output
├── prompts.py            # Default prompt templates
├── types.py              # Type definitions and data classes
├── utils.py              # Utility functions
└── README.md             # This file
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Type Hints**: Comprehensive type annotations for better IDE support and code quality
- **Retry Logic**: Automatic retry with exponential backoff for retriable errors
- **Structured Output**: JSON schema validation for consistent responses
- **Performance Metrics**: Detailed tracking of token usage, timing, and costs
- **Markdown Logging**: Comprehensive logs of all API requests and responses
- **Database Management**: Context manager for safe database operations

## Components

### Core Classes

#### `MaterialsDescriptionGenerator`
Main class for generating descriptions. Orchestrates the entire process.

```python
from pdf_description_gen import MaterialsDescriptionGenerator

generator = MaterialsDescriptionGenerator()
generator.process_materials_db()
```

#### `APIClient`
Handles API communication with automatic retry logic for transient errors.

#### `DatabaseManager`
Manages SQLite database operations with context manager support.

```python
from pdf_description_gen import DatabaseManager

with DatabaseManager(db_path) as db:
    rows = db.fetch_all_materials()
    # ... process rows ...
    db.update_materials_batch(rows)
    db.commit()
```

#### `MarkdownLogger`
Logs API requests and responses to markdown files for debugging.

### Data Types

All data types are defined in `types.py`:

- `MaterialRow`: Database row representation
- `PerformanceMetrics`: Timing and cost metrics
- `GenerationResult`: API response with metrics
- `TaskType`: Enum for task types (INDIVIDUAL/BATCH)
- `BatchInfo`: Information for batch processing
- `ProcessingTask`: Unified task representation

### Schemas

JSON schemas for structured output:
- `get_single_page_schema()`: Schema for individual page analysis
- `get_batch_schema()`: Schema for batch analysis

### Prompts

Default prompts are defined in `prompts.py` and can be overridden via:
1. Function arguments
2. Configuration file (`config.yml`)
3. Built-in defaults

## Configuration

The package reads configuration from `config.yml` via `config_loader`:

```yaml
api:
  platform: openai  # or other compatible platforms
  openai:
    base_url: https://api.openai.com/v1
    key: your-api-key
    vlm_model: gpt-4o
    service_tier: auto
  timeout: 3600

paths:
  materials_db_path: materials/processed_materials.db
  materials_dir: materials/processed

batch:
  size: 10

prompts:
  vlm:
    single: "Your custom single page prompt..."
    batch: "Your custom batch prompt..."
```

## Usage Examples

### Basic Usage

```python
from pdf_description_gen import MaterialsDescriptionGenerator

# Initialize with defaults from config.yml
generator = MaterialsDescriptionGenerator()

# Process all materials
generator.process_materials_db()
```

### Single Image Description

```python
generator = MaterialsDescriptionGenerator()

result = generator.generate_description(
    image_path="path/to/image.jpg",
    prompt="Describe this educational content",
    return_metrics=True
)

print(f"Description: {result.content}")
print(f"Cost: ${result.metrics.total_cost:.6f}")
```

### Custom Configuration

```python
generator = MaterialsDescriptionGenerator(
    api_base="https://custom-api.example.com/v1",
    api_key="custom-key",
    model_name="custom-model",
    timeout=1800
)
```

### Direct Database Operations

```python
from pdf_description_gen import DatabaseManager

with DatabaseManager("materials/processed_materials.db") as db:
    materials = db.fetch_all_materials()
    
    for material in materials:
        if material.status == "pending":
            print(f"Processing: {material.original_filename}")
```

## Error Handling

The package automatically retries on:
- HTTP 500, 502, 503, 504 errors
- Connection errors
- Timeout errors

With exponential backoff: 2s → 5s → 10s → 15s → 30s (max)

Non-retriable errors (e.g., 400, 401, 403) fail immediately.

## Performance Tracking

All API calls collect detailed metrics:
- Token counts (prompt, completion, total)
- Timing (total time, TTFT, generation time)
- Throughput (tokens/sec for prompt and generation)
- Costs (input, output, total)

Metrics are logged to markdown files and summarized at the end.

## Development

The refactored structure provides:
- **Better testability**: Each module can be tested independently
- **Easier maintenance**: Changes are localized to specific modules
- **Type safety**: Comprehensive type hints catch errors early
- **Clearer responsibilities**: Each class has a single, well-defined purpose

## Migration from Original

The new package maintains the same public API:

```python
# Old
from pdf_description_gen import MaterialsDescriptionGenerator
generator = MaterialsDescriptionGenerator()
generator.process_materials_db()

# New (same interface!)
from pdf_description_gen import MaterialsDescriptionGenerator
generator = MaterialsDescriptionGenerator()
generator.process_materials_db()
```

Internal implementation is now modular and maintainable.

