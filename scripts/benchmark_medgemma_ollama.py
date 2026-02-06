"""
Comprehensive MedGemma Benchmark (Ollama Version)
Runs all benchmarks: MedQA, PubMedQA, Vietnamese Medical QA
Uses MedGemma-Q8 via Ollama (fast local inference)
"""
import json
import time
import random
import httpx
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
MODEL = "medgemma-q8"

def query_ollama(prompt: str, max_tokens: int = 128) -> tuple[str, float]:
    """Query MedGemma via Ollama"""
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

def benchmark_medqa(n_samples: int = 50):
    """Benchmark on MedQA (USMLE)"""
    print("\n" + "="*60)
    print("BENCHMARK: MedQA (USMLE)")
    print("="*60)
    
    with open(DATASETS_DIR / "medqa_test.json", "r") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    correct = 0
    total_time = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        options = sample["options"]
        answer_idx = sample["answer_idx"]  # This is the letter (A, B, C, D, E)
        
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        # More explicit prompt to get just the letter
        prompt = f"""You are a medical expert taking the USMLE exam. Answer with ONLY the letter of the correct answer.

Question: {question}

Options:
{options_text}

Answer (one letter only):"""
        
        response, gen_time = query_ollama(prompt, max_tokens=20)
        total_time += gen_time
        
        # Extract answer letter from response
        pred = ""
        response_upper = response.upper().strip()
        for char in response_upper:
            if char in "ABCDE":
                pred = char
                break
        
        is_correct = pred == answer_idx
        if is_correct:
            correct += 1
        
        print(f"[{i+1}/{n_samples}] {'✅' if is_correct else '❌'} Pred: {pred}, Truth: {answer_idx} ({gen_time:.1f}s)")
    
    accuracy = 100 * correct / n_samples
    avg_time = total_time / n_samples
    print(f"\nMedQA Results: {correct}/{n_samples} ({accuracy:.1f}%) | Avg: {avg_time:.1f}s")
    return {"dataset": "MedQA", "accuracy": accuracy, "correct": correct, "total": n_samples, "avg_time": avg_time}

def benchmark_pubmedqa(n_samples: int = 50):
    """Benchmark on PubMedQA"""
    print("\n" + "="*60)
    print("BENCHMARK: PubMedQA")
    print("="*60)
    
    with open(DATASETS_DIR / "pubmedqa_test.json", "r") as f:
        data = json.load(f)  # This is a list
    
    samples = random.sample(data, min(n_samples, len(data)))
    correct = 0
    total_time = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        context = sample["context"][:1200]  # Truncate context
        answer = sample["final_decision"]
        
        prompt = f"""Based on this research, answer the question with ONLY 'yes', 'no', or 'maybe'.

Context: {context}

Question: {question}

Answer (yes/no/maybe):"""
        
        response, gen_time = query_ollama(prompt, max_tokens=20)
        total_time += gen_time
        
        pred = response.lower().strip()
        if "yes" in pred[:15]:
            pred = "yes"
        elif "no" in pred[:15]:
            pred = "no"
        else:
            pred = "maybe"
        
        is_correct = pred == answer
        if is_correct:
            correct += 1
        
        print(f"[{i+1}/{n_samples}] {'✅' if is_correct else '❌'} Pred: {pred}, Truth: {answer} ({gen_time:.1f}s)")
    
    accuracy = 100 * correct / n_samples
    avg_time = total_time / n_samples
    print(f"\nPubMedQA Results: {correct}/{n_samples} ({accuracy:.1f}%) | Avg: {avg_time:.1f}s")
    return {"dataset": "PubMedQA", "accuracy": accuracy, "correct": correct, "total": n_samples, "avg_time": avg_time}

def benchmark_vietnamese(n_samples: int = 30):
    """Benchmark on Vietnamese Medical QA"""
    print("\n" + "="*60)
    print("BENCHMARK: Vietnamese Medical QA")
    print("="*60)
    
    with open(DATASETS_DIR / "vi_medqa.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    total_time = 0
    results = []
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        reference = sample["answer"][:150]
        
        prompt = f"""Bạn là bác sĩ chuyên khoa. Trả lời ngắn gọn (2-3 câu) bằng tiếng Việt.

Câu hỏi: {question}

Trả lời:"""
        
        response, gen_time = query_ollama(prompt, max_tokens=150)
        total_time += gen_time
        
        results.append({
            "question": question[:80],
            "reference": reference,
            "prediction": response[:150],
            "time": gen_time
        })
        
        print(f"[{i+1}/{n_samples}] Q: {question[:40]}... ({gen_time:.1f}s)")
        print(f"   A: {response[:80]}...")
    
    avg_time = total_time / n_samples
    print(f"\nVietnamese QA: {n_samples} samples | Avg: {avg_time:.1f}s")
    return {"dataset": "Vietnamese", "samples": n_samples, "avg_time": avg_time, "results": results}

def run_all_benchmarks():
    """Run all benchmarks"""
    print("="*60)
    print("COMPREHENSIVE MedGemma BENCHMARK (Ollama)")
    print(f"Model: {MODEL}")
    print("="*60)
    
    results = []
    results.append(benchmark_medqa(n_samples=50))
    results.append(benchmark_pubmedqa(n_samples=50))
    results.append(benchmark_vietnamese(n_samples=30))
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for r in results:
        if "accuracy" in r:
            print(f"{r['dataset']}: {r['accuracy']:.1f}% ({r['correct']}/{r['total']}) | {r['avg_time']:.1f}s avg")
        else:
            print(f"{r['dataset']}: {r['samples']} samples | {r['avg_time']:.1f}s avg (qualitative)")
    
    # Save results
    output_file = DATASETS_DIR / "medgemma_ollama_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)
    run_all_benchmarks()
