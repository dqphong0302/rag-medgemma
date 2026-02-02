# RAG-MedGemma: Clinical Decision Support System

A GraphRAG-powered clinical decision support system using MedGemma for EdgeAI deployment (Intel NUC).

## Project Structure

```
medgemma/
├── data/
│   ├── input/          # PubMed documents for RAG
│   ├── datasets/       # MedQA, PubMedQA, VQA-RAD, PathVQA
│   ├── output/         # GraphRAG indexed data
│   └── settings.yaml   # GraphRAG configuration
├── scripts/
│   ├── download_datasets.py  # Download medical QA datasets
│   └── benchmark.py          # Evaluation benchmark
├── src/
│   ├── app.py          # FastAPI backend
│   └── static/         # Web UI
└── models/             # GGUF model files (not in git)
```

## Setup

### 1. Install Ollama

```powershell
winget install Ollama.Ollama
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

### 2. Create Conda Environment

```powershell
conda create -n medgemma_env python=3.10
conda activate medgemma_env
pip install -r requirements.txt
```

### 3. Download Datasets

```powershell
python scripts/download_datasets.py
```

### 4. Index Documents

```powershell
graphrag index --root data
```

### 5. Run Application

```powershell
cd src
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Datasets

| Dataset | Type | Size |
|---------|------|------|
| MedQA-USMLE | Text QA | 1,273 questions |
| PubMedQA | Text QA | 1,000 questions |
| VQA-RAD | Image QA | ~451 samples |
| PathVQA | Image QA | 500 samples |

## Benchmark Results

| Model | MedQA | PubMedQA | Latency |
|-------|-------|----------|---------|
| gemma2:2b (baseline) | 34% | TBD | ~3.3s |
| gemma2:2b + RAG | TBD | TBD | TBD |

## License

MIT
