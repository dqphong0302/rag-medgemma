"""
Benchmark Evaluation Script for RAG-MedGemma
Compares baseline LLM vs RAG-enhanced performance on medical QA datasets
"""
import json
import time
import httpx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Paths
DATA_DIR = Path("d:/medgemma/data")
DATASETS_DIR = DATA_DIR / "datasets"
RESULTS_DIR = DATA_DIR / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ollama config
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "gemma2:2b"

# CDS API (for RAG)
CDS_URL = "http://localhost:8080"

def query_ollama_direct(question: str, options: list = None) -> tuple[str, float]:
    """Query Ollama directly (no RAG) - baseline"""
    start = time.time()
    
    if options:
        # Handle both dict {'A': '...', 'B': '...'} and list formats
        if isinstance(options, dict):
            opts = [options.get('A', ''), options.get('B', ''), options.get('C', ''), options.get('D', '')]
        else:
            opts = options
        prompt = f"""You are a medical expert. Answer this USMLE question by selecting the correct option.

Question: {question}

Options:
A. {opts[0]}
B. {opts[1]}
C. {opts[2]}
D. {opts[3]}

Reply with ONLY the letter (A, B, C, or D) of the correct answer."""
    else:
        prompt = f"""You are a medical expert. Answer this question with yes, no, or maybe.

Question: {question}

Reply with ONLY one word: yes, no, or maybe."""
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
        )
    
    latency = time.time() - start
    answer = response.json().get("response", "").strip().upper()
    return answer, latency

def query_rag(question: str) -> tuple[str, float]:
    """Query via RAG-enhanced CDS API"""
    start = time.time()
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{CDS_URL}/query",
            json={"question": question}
        )
    
    latency = time.time() - start
    if response.status_code == 200:
        return response.json().get("answer", ""), latency
    return "", latency

def benchmark_medqa(limit: int = 50):
    """Benchmark on MedQA-USMLE dataset"""
    print("\n" + "="*60)
    print("Benchmarking MedQA-USMLE (Multiple Choice)")
    print("="*60)
    
    with open(DATASETS_DIR / "medqa_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:limit]
    
    results = []
    correct = 0
    total_latency = 0
    
    for item in tqdm(data, desc="MedQA"):
        question = item["question"]
        options = item["options"]
        answer_idx = item["answer_idx"]
        # Handle both letter (A/B/C/D) and number (0/1/2/3) formats
        if isinstance(answer_idx, str) and answer_idx in "ABCD":
            correct_idx = ord(answer_idx) - ord('A')
        else:
            correct_idx = int(answer_idx)
        correct_letter = ["A", "B", "C", "D"][correct_idx]
        
        answer, latency = query_ollama_direct(question, options)
        
        # Extract first letter
        pred_letter = answer[0] if answer else ""
        is_correct = pred_letter == correct_letter
        
        if is_correct:
            correct += 1
        total_latency += latency
        
        results.append({
            "question": question[:100],
            "correct": correct_letter,
            "predicted": pred_letter,
            "is_correct": is_correct,
            "latency_s": round(latency, 2)
        })
    
    accuracy = correct / len(data) * 100
    avg_latency = total_latency / len(data)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(data)})")
    print(f"  Avg Latency: {avg_latency:.2f}s")
    
    return {
        "dataset": "MedQA-USMLE",
        "samples": len(data),
        "accuracy": round(accuracy, 2),
        "avg_latency_s": round(avg_latency, 2),
        "details": results
    }

def benchmark_pubmedqa(limit: int = 50):
    """Benchmark on PubMedQA dataset"""
    print("\n" + "="*60)
    print("Benchmarking PubMedQA (Yes/No/Maybe)")
    print("="*60)
    
    with open(DATASETS_DIR / "pubmedqa_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:limit]
    
    results = []
    correct = 0
    total_latency = 0
    
    for item in tqdm(data, desc="PubMedQA"):
        question = item["question"]
        expected = item["final_decision"].lower()  # yes/no/maybe
        
        answer, latency = query_ollama_direct(question)
        
        # Normalize answer
        answer_lower = answer.lower()
        if "yes" in answer_lower:
            pred = "yes"
        elif "no" in answer_lower:
            pred = "no"
        elif "maybe" in answer_lower:
            pred = "maybe"
        else:
            pred = answer_lower[:10]
        
        is_correct = pred == expected
        if is_correct:
            correct += 1
        total_latency += latency
        
        results.append({
            "question": question[:100],
            "expected": expected,
            "predicted": pred,
            "is_correct": is_correct,
            "latency_s": round(latency, 2)
        })
    
    accuracy = correct / len(data) * 100
    avg_latency = total_latency / len(data)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(data)})")
    print(f"  Avg Latency: {avg_latency:.2f}s")
    
    return {
        "dataset": "PubMedQA",
        "samples": len(data),
        "accuracy": round(accuracy, 2),
        "avg_latency_s": round(avg_latency, 2),
        "details": results
    }

def main():
    print("="*60)
    print("RAG-MedGemma Benchmark Evaluation")
    print(f"Model: {LLM_MODEL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Run benchmarks
    medqa_results = benchmark_medqa(limit=50)
    pubmedqa_results = benchmark_pubmedqa(limit=50)
    
    # Save results
    results = {
        "model": LLM_MODEL,
        "timestamp": datetime.now().isoformat(),
        "rag_enabled": False,
        "benchmarks": [medqa_results, pubmedqa_results]
    }
    
    output_file = RESULTS_DIR / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (Baseline - No RAG)")
    print("="*60)
    print(f"{'Dataset':<20} {'Accuracy':<15} {'Latency':<15}")
    print("-"*50)
    print(f"{'MedQA-USMLE':<20} {medqa_results['accuracy']:.1f}%{'':<10} {medqa_results['avg_latency_s']:.2f}s")
    print(f"{'PubMedQA':<20} {pubmedqa_results['accuracy']:.1f}%{'':<10} {pubmedqa_results['avg_latency_s']:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
