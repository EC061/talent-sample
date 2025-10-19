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