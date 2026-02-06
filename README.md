# 🏥 RAG-MedGemma: EdgeAI Clinical Decision Support System

> **A Privacy-First, Offline-Capable Clinical Decision Support System for Vietnamese Hospitals**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: MedGemma](https://img.shields.io/badge/Model-MedGemma--Q8-blue)](https://ollama.com)
[![Status: Optimized](https://img.shields.io/badge/Status-Optimized-green)]()

## 🌟 Key Features

* **Offline First:** Runs 100% locally on Edge devices (Laptop/MiniPC with 16GB RAM).
* **Privacy Preserved:** No patient data is sent to the cloud.
* **Cost Effective:** Zero inference cost compared to Cloud APIs.
* **Pure Knowledge RAG:** Optimized retrieval system achieving **59% accuracy** on MedQA (+4% vs baseline).

---

## 🔑 Key Research Finding: "Format > Size"

Our experiments revealed that the **format of the knowledge base matters more than its size**.

| RAG Approach | Knowledge Base | Accuracy (MedQA) | Impact |
|--------------|----------------|------------------|--------|
| **Baseline** | None | 55.0% | - |
| **Q&A Format** | 10,178 Q&A pairs (14.9 MB) | 50.0% | **-5.0%** ❌ |
| **Pure Knowledge** | 20,336 facts (29.8 MB) | **59.0%** | **+4.0%** ✅ |

> **Why?** Q&A pairs confuse the retriever with "similar questions," whereas Pure Knowledge provides clean facts that support the LLM's reasoning.

---

## 🛠️ System Architecture

```mermaid
graph LR
    User[Doctor/User] --> Query
    Query --> Retriever[Pure Knowledge Retriever]
    Retriever --> KB[(Pure Medical Facts DB)]
    KB --> Context[Relevant Facts]
    Context --> LLM[MedGemma-Q8 (Edge)]
    LLM --> Answer
```

### Technical Specs

* **LLM:** MedGemma-4B-IT (Quantized Q8_0) - 4.13 GB
* **Embedding:** all-MiniLM-L6-v2 (384-dim)
* **Vector DB:** FAISS IndexFlatIP
* **Runtime:** Ollama (Local Inference)

---

## 📊 Benchmark Results

### 1. MedQA (USMLE)

* **Edge (Proprietary RAG):** 59.0%
* **Cloud (GPT-4):** ~85%
* *Trade-off:* Edge offers privacy and offline capability with acceptable accuracy for screening support.

### 2. VQA-RAD (Medical Imaging)

* Successfully processes X-ray, CT, and MRI images.
* **Accuracy:** 100% on initial test set (5 samples).

---

## 🚀 Installation & Setup

### Prerequisites

* OS: macOS / Linux / Windows WSL2
* RAM: 16GB minimum
* Python: 3.10+

### 1. Install Ollama & Pull Model

```bash
# Install Ollama from ollama.com
ollama pull medgemma-q8  # Custom model (or import Modelfile)
ollama pull nomic-embed-text
```

### 2. Environment Setup

```bash
conda create -n medgemma_env python=3.10
conda activate medgemma_env
pip install -r requirements.txt
```

### 3. Initialize Knowledge Base

```bash
# Create Pure Knowledge Index
python scripts/create_knowledge_base.py
```

### 4. Run API Server

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8081
```

---

## 📑 Presentation & Reports

We have prepared comprehensive documentation for the conference:

* **📄 [Conference Report](conference_report_vi.md):** Full academic report (Vietnamese).
* **🖼️ [Presentation Slides](presentation/index.html):** 20-slide interactive presentation with visualizations.
* **🎤 [Presentation Script](presentation/present.md):** Detailed oral presentation script.

To view the slides:

```bash
open presentation/index.html
```

---

## 📂 Project Structure

```
rag-medgemma/
├── data/
│   ├── datasets/       # MedQA, PubMedQA, etc.
│   └── output/         # FAISS indexes (Pure Knowledge)
├── presentation/       # Slides and images
│   ├── images/
│   ├── index.html
│   └── present.md
├── scripts/            # Benchmarks & Indexing tools
│   ├── benchmark_pure_knowledge_rag.py
│   ├── create_knowledge_base.py
│   └── ...
├── src/
│   ├── app.py          # API Server
│   └── rag_engine.py   # RAG Logic
└── README.md
```

## License

MIT
