"""
Pure Knowledge RAG Benchmark
Uses medical facts instead of Q&A format
"""
import json
import time
import random
import httpx
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
OUTPUT_DIR = DATA_DIR / "output"

OLLAMA_URL = "http://localhost:11434"
MODEL = "medgemma-q8"

class PureKnowledgeRetriever:
    """Retriever using pure medical facts (no Q&A format)"""
    
    def __init__(self):
        print("Loading Pure Knowledge Retriever...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.index = faiss.read_index(str(OUTPUT_DIR / "medical_knowledge_index.faiss"))
        with open(OUTPUT_DIR / "medical_knowledge_docs.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        print(f"Loaded {len(self.documents)} medical knowledge documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> str:
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        context_parts = ["MEDICAL KNOWLEDGE:"]
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score > 0.35:
                doc = self.documents[idx]
                context_parts.append(f"\n• {doc}")
        
        return "\n".join(context_parts) if len(context_parts) > 1 else ""

def query_ollama(prompt: str, max_tokens: int = 128) -> tuple[str, float]:
    start_time = time.time()
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1}
            }
        )
    gen_time = time.time() - start_time
    return response.json().get("response", "").strip(), gen_time

def run_benchmark(retriever, n_samples: int = 100):
    print(f"\n{'='*60}")
    print(f"MedQA BENCHMARK: {n_samples} samples with Pure Knowledge RAG")
    print("="*60)
    
    with open(DATASETS_DIR / "medqa_test.json", "r") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    baseline_correct = 0
    rag_correct = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        options = sample["options"]
        answer_idx = sample["answer_idx"]
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        # Baseline
        prompt_baseline = f"""Answer with ONLY the letter (A-E).

Question: {question}

Options:
{options_text}

Answer:"""
        
        response_baseline, _ = query_ollama(prompt_baseline, max_tokens=10)
        pred_baseline = ""
        for char in response_baseline.upper():
            if char in "ABCDE":
                pred_baseline = char
                break
        
        # With Pure Knowledge RAG
        context = retriever.retrieve(question, top_k=5)
        
        if context and len(context) > 50:
            prompt_rag = f"""Use this medical knowledge. Answer with ONLY the letter (A-E).

{context}

Question: {question}

Options:
{options_text}

Answer:"""
        else:
            prompt_rag = prompt_baseline
        
        response_rag, _ = query_ollama(prompt_rag, max_tokens=10)
        pred_rag = ""
        for char in response_rag.upper():
            if char in "ABCDE":
                pred_rag = char
                break
        
        if pred_baseline == answer_idx:
            baseline_correct += 1
        if pred_rag == answer_idx:
            rag_correct += 1
        
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{n_samples}] Baseline: {baseline_correct}/{i+1} ({100*baseline_correct/(i+1):.1f}%) | +RAG: {rag_correct}/{i+1} ({100*rag_correct/(i+1):.1f}%)")
    
    baseline_acc = 100 * baseline_correct / n_samples
    rag_acc = 100 * rag_correct / n_samples
    
    print(f"\n{'='*40}")
    print(f"RESULTS ({n_samples} samples):")
    print(f"  Baseline:       {baseline_correct}/{n_samples} ({baseline_acc:.2f}%)")
    print(f"  +Pure Knowledge: {rag_correct}/{n_samples} ({rag_acc:.2f}%)")
    print(f"  Improvement:    {rag_acc - baseline_acc:+.2f}%")
    
    return {
        "n_samples": n_samples,
        "baseline_accuracy": baseline_acc,
        "rag_accuracy": rag_acc,
        "improvement": rag_acc - baseline_acc
    }

def main():
    start = datetime.now()
    print("="*60)
    print("PURE KNOWLEDGE RAG BENCHMARK")
    print(f"Using: 20,336 medical fact documents")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    retriever = PureKnowledgeRetriever()
    results = run_benchmark(retriever, n_samples=100)
    
    end = datetime.now()
    duration = (end - start).total_seconds()
    
    print(f"\nTotal duration: {duration/60:.1f} minutes")
    
    results["duration_seconds"] = duration
    output_file = DATASETS_DIR / "pure_knowledge_rag_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)
    main()
