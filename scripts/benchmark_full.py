"""
Maximum Sample Benchmark - Full Dataset Run
MedQA: 200 samples, PubMedQA: 200 samples, Vietnamese: 100 samples
With Hybrid RAG comparison
"""
import json
import time
import random
import httpx
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from hybrid_retriever import HybridRetriever

DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"

OLLAMA_URL = "http://localhost:11434"
MODEL = "medgemma-q8"

# Initialize retriever
print("Loading HybridRetriever...")
retriever = HybridRetriever(DATA_DIR, language="en")

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

def run_medqa(n_samples: int = 200):
    """Full MedQA benchmark with RAG comparison"""
    print(f"\n{'='*60}")
    print(f"MedQA BENCHMARK: {n_samples} samples")
    print("="*60)
    
    with open(DATASETS_DIR / "medqa_test.json", "r") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    baseline_correct = 0
    rag_correct = 0
    total_time = 0
    errors = []
    
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
        
        response_baseline, t = query_ollama(prompt_baseline, max_tokens=10)
        total_time += t
        
        pred_baseline = ""
        for char in response_baseline.upper():
            if char in "ABCDE":
                pred_baseline = char
                break
        
        # With RAG
        context, _ = retriever.retrieve(question, top_k_semantic=5, top_k_entities=5, min_score=0.2)
        
        if context:
            prompt_rag = f"""Use this medical knowledge to help answer. Answer with ONLY the letter (A-E).

{context[:1500]}

Question: {question}

Options:
{options_text}

Answer:"""
        else:
            prompt_rag = prompt_baseline
        
        response_rag, t = query_ollama(prompt_rag, max_tokens=10)
        total_time += t
        
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
        
        # Collect errors for analysis
        if not baseline_ok:
            errors.append({"idx": i, "question": question[:100], "pred": pred_baseline, "truth": answer_idx})
        
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{n_samples}] Baseline: {baseline_correct}/{i+1} ({100*baseline_correct/(i+1):.1f}%) | RAG: {rag_correct}/{i+1} ({100*rag_correct/(i+1):.1f}%)")
    
    baseline_acc = 100 * baseline_correct / n_samples
    rag_acc = 100 * rag_correct / n_samples
    
    print(f"\n{'='*40}")
    print(f"MedQA FINAL RESULTS ({n_samples} samples):")
    print(f"  Baseline: {baseline_correct}/{n_samples} ({baseline_acc:.2f}%)")
    print(f"  +RAG:     {rag_correct}/{n_samples} ({rag_acc:.2f}%)")
    print(f"  Diff:     {rag_acc - baseline_acc:+.2f}%")
    print(f"  Avg time: {total_time/n_samples/2:.2f}s per query")
    
    return {
        "dataset": "MedQA",
        "n_samples": n_samples,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_acc,
        "rag_correct": rag_correct,
        "rag_accuracy": rag_acc,
        "improvement": rag_acc - baseline_acc,
        "avg_time": total_time / n_samples / 2,
        "error_samples": errors[:10]  # First 10 errors
    }

def run_pubmedqa(n_samples: int = 200):
    """Full PubMedQA benchmark"""
    print(f"\n{'='*60}")
    print(f"PubMedQA BENCHMARK: {n_samples} samples")
    print("="*60)
    
    with open(DATASETS_DIR / "pubmedqa_test.json", "r") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    correct = 0
    total_time = 0
    distribution = {"yes": 0, "no": 0, "maybe": 0}
    pred_distribution = {"yes": 0, "no": 0, "maybe": 0}
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        context = sample["context"][:800]
        answer = sample["final_decision"]
        distribution[answer] = distribution.get(answer, 0) + 1
        
        prompt = f"""Answer ONLY: yes, no, or maybe.

Context: {context}

Question: {question}

Answer:"""
        
        response, t = query_ollama(prompt, max_tokens=10)
        total_time += t
        
        pred = response.lower().strip()
        if "yes" in pred[:15]:
            pred = "yes"
        elif "no" in pred[:15]:
            pred = "no"
        else:
            pred = "maybe"
        
        pred_distribution[pred] = pred_distribution.get(pred, 0) + 1
        
        if pred == answer:
            correct += 1
        
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{n_samples}] Correct: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")
    
    accuracy = 100 * correct / n_samples
    
    print(f"\n{'='*40}")
    print(f"PubMedQA FINAL RESULTS ({n_samples} samples):")
    print(f"  Accuracy: {correct}/{n_samples} ({accuracy:.2f}%)")
    print(f"  Avg time: {total_time/n_samples:.2f}s")
    print(f"  Ground truth distribution: {distribution}")
    print(f"  Prediction distribution:   {pred_distribution}")
    
    return {
        "dataset": "PubMedQA",
        "n_samples": n_samples,
        "correct": correct,
        "accuracy": accuracy,
        "avg_time": total_time / n_samples,
        "gt_distribution": distribution,
        "pred_distribution": pred_distribution
    }

def run_vietnamese(n_samples: int = 100):
    """Vietnamese QA benchmark"""
    print(f"\n{'='*60}")
    print(f"VIETNAMESE QA BENCHMARK: {n_samples} samples")
    print("="*60)
    
    with open(DATASETS_DIR / "vi_medqa.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    total_time = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        
        prompt = f"""Bạn là bác sĩ. Trả lời ngắn gọn (2-3 câu) bằng tiếng Việt.

Câu hỏi: {question}

Trả lời:"""
        
        response, t = query_ollama(prompt, max_tokens=150)
        total_time += t
        
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{n_samples}] Avg time: {total_time/(i+1):.1f}s")
    
    avg_time = total_time / n_samples
    
    print(f"\n{'='*40}")
    print(f"Vietnamese QA RESULTS ({n_samples} samples):")
    print(f"  Avg time: {avg_time:.2f}s")
    
    return {
        "dataset": "Vietnamese",
        "n_samples": n_samples,
        "avg_time": avg_time,
        "note": "Qualitative evaluation"
    }

def main():
    start = datetime.now()
    print("="*60)
    print("COMPREHENSIVE MedGemma Q8 + Hybrid RAG BENCHMARK")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL}")
    print("="*60)
    
    results = {}
    
    # Run all benchmarks
    results["medqa"] = run_medqa(n_samples=200)
    results["pubmedqa"] = run_pubmedqa(n_samples=200)
    results["vietnamese"] = run_vietnamese(n_samples=100)
    
    end = datetime.now()
    duration = (end - start).total_seconds()
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL COMPREHENSIVE SUMMARY")
    print("="*60)
    print(f"\nMedQA ({results['medqa']['n_samples']} samples):")
    print(f"  Baseline: {results['medqa']['baseline_accuracy']:.2f}%")
    print(f"  +RAG:     {results['medqa']['rag_accuracy']:.2f}%")
    print(f"  Improve:  {results['medqa']['improvement']:+.2f}%")
    
    print(f"\nPubMedQA ({results['pubmedqa']['n_samples']} samples):")
    print(f"  Accuracy: {results['pubmedqa']['accuracy']:.2f}%")
    
    print(f"\nVietnamese ({results['vietnamese']['n_samples']} samples):")
    print(f"  Avg time: {results['vietnamese']['avg_time']:.2f}s")
    
    print(f"\nTotal duration: {duration/60:.1f} minutes")
    print(f"Completed: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    results["metadata"] = {
        "model": MODEL,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": duration
    }
    
    output_file = DATASETS_DIR / "full_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)
    main()
