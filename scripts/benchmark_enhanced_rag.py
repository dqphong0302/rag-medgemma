"""
Enhanced RAG Benchmark using MedQA Training Index
Uses 10,178 USMLE Q&A documents for better domain-specific retrieval
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

class EnhancedRetriever:
    """Retriever using MedQA training index for domain-specific context"""
    
    def __init__(self):
        print("Loading Enhanced Retriever with MedQA training data...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load MedQA training index
        self.index = faiss.read_index(str(OUTPUT_DIR / "medqa_training_index.faiss"))
        with open(OUTPUT_DIR / "medqa_training_docs.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        print(f"Loaded {len(self.documents)} USMLE documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant USMLE documents for query"""
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        context_parts = ["RELEVANT USMLE KNOWLEDGE:"]
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score > 0.3:  # Minimum relevance threshold
                doc = self.documents[idx]
                context_parts.append(f"\n[{i+1}] (similarity: {score:.2f})\n{doc[:600]}")
        
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

def run_medqa_benchmark(retriever, n_samples: int = 100):
    """Benchmark MedQA with enhanced RAG"""
    print(f"\n{'='*60}")
    print(f"MedQA BENCHMARK: {n_samples} samples with Enhanced RAG")
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
        
        # With Enhanced RAG
        context = retriever.retrieve(question, top_k=3)
        
        if context:
            prompt_rag = f"""Use this USMLE knowledge to help answer. Answer with ONLY the letter (A-E).

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
        
        baseline_ok = pred_baseline == answer_idx
        rag_ok = pred_rag == answer_idx
        
        if baseline_ok:
            baseline_correct += 1
        if rag_ok:
            rag_correct += 1
        
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{n_samples}] Baseline: {baseline_correct}/{i+1} ({100*baseline_correct/(i+1):.1f}%) | +RAG: {rag_correct}/{i+1} ({100*rag_correct/(i+1):.1f}%)")
    
    baseline_acc = 100 * baseline_correct / n_samples
    rag_acc = 100 * rag_correct / n_samples
    
    print(f"\n{'='*40}")
    print(f"MedQA RESULTS ({n_samples} samples):")
    print(f"  Baseline:     {baseline_correct}/{n_samples} ({baseline_acc:.2f}%)")
    print(f"  +Enhanced RAG: {rag_correct}/{n_samples} ({rag_acc:.2f}%)")
    print(f"  Improvement:  {rag_acc - baseline_acc:+.2f}%")
    
    return {
        "dataset": "MedQA",
        "n_samples": n_samples,
        "baseline_accuracy": baseline_acc,
        "rag_accuracy": rag_acc,
        "improvement": rag_acc - baseline_acc
    }

def main():
    start = datetime.now()
    print("="*60)
    print("ENHANCED RAG BENCHMARK")
    print(f"Using: MedQA Training Index (10,178 USMLE documents)")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    retriever = EnhancedRetriever()
    
    # Run benchmark with 100 samples
    results = run_medqa_benchmark(retriever, n_samples=100)
    
    end = datetime.now()
    duration = (end - start).total_seconds()
    
    print(f"\nTotal duration: {duration/60:.1f} minutes")
    
    # Save results
    results["duration_seconds"] = duration
    output_file = DATASETS_DIR / "enhanced_rag_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)
    main()
