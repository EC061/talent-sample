## Project Description

Minimal toolkit for turning PDFs into images and generating page- and document-level descriptions with a vision LLM. Data is tracked in `materials/processed_materials.db`, and the model can be served locally via an OpenAI-compatible API using `vLLM` (see deployment below).

## Setup

```bash
sudo apt update && sudo apt install -y poppler-utils
uv venv --python=3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# optional: needed for the first model download
huggingface-cli login
```

## Deployment (OpenAI-compatible API via vLLM)

```bash
# Start with defaults (reads config from config.yml if present)
./start_server

# Custom port
./start_server --port 9000

# Choose GPUs and tensor parallelism
./start_server --cuda-devices "0,1" -tp 2

# Override model
./start_server --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

The API will be available at `http://HOST:PORT/v1` (defaults: `0.0.0.0:8000`).