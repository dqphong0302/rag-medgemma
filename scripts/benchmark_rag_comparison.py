"""
Benchmark: MedGemma Baseline vs Hybrid RAG comparison
Compare performance with and without RAG context
"""
import json
import time
import random
import httpx
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"

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

def load_rag_context(question: str) -> str:
    """Load relevant RAG context from indexed documents"""
    # Simple keyword-based retrieval from medical content
    rag_file = DATA_DIR / "output" / "input" / "create_final_documents.parquet"
    
    if rag_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(rag_file)
            # Simple keyword search
            keywords = question.lower().split()[:5]
            relevant = []
            for _, row in df.iterrows():
                text = str(row.get('text', '')).lower()
                if any(kw in text for kw in keywords):
                    relevant.append(str(row.get('text', ''))[:300])
                    if len(relevant) >= 2:
                        break
            return "\n".join(relevant)
        except:
            pass
    
    # Fallback: load from text units
    text_units_file = DATA_DIR / "output" / "input" / "create_base_text_units.parquet"
    if text_units_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(text_units_file)
            # Simple keyword search
            keywords = question.lower().split()[:5]
            relevant = []
            for _, row in df.iterrows():
                text = str(row.get('text', '')).lower()
                if any(kw in text for kw in keywords):
                    relevant.append(str(row.get('text', ''))[:300])
                    if len(relevant) >= 2:
                        break
            return "\n".join(relevant)
        except:
            pass
    return ""

def benchmark_medqa_comparison(n_samples: int = 30):
    """Compare Baseline vs RAG on MedQA"""
    print("\n" + "="*60)
    print("BENCHMARK: MedQA - Baseline vs RAG")
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
        
        # Baseline (no RAG)
        prompt_baseline = f"""Answer with ONLY the letter (A-E).

Question: {question}

Options:
{options_text}

Answer:"""
        
        response_baseline, time_baseline = query_ollama(prompt_baseline, max_tokens=10)
        pred_baseline = ""
        for char in response_baseline.upper():
            if char in "ABCDE":
                pred_baseline = char
                break
        
        # With RAG context
        context = load_rag_context(question)
        if context:
            prompt_rag = f"""Use the context to help answer. Answer with ONLY the letter (A-E).

Context: {context[:500]}

Question: {question}

Options:
{options_text}

Answer:"""
        else:
            prompt_rag = prompt_baseline
        
        response_rag, time_rag = query_ollama(prompt_rag, max_tokens=10)
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
        
        print(f"[{i+1}/{n_samples}] Base: {'✅' if baseline_ok else '❌'}{pred_baseline} | RAG: {'✅' if rag_ok else '❌'}{pred_rag} | Truth: {answer_idx}")
    
    baseline_acc = 100 * baseline_correct / n_samples
    rag_acc = 100 * rag_correct / n_samples
    
    print(f"\n{'='*40}")
    print(f"Baseline: {baseline_correct}/{n_samples} ({baseline_acc:.1f}%)")
    print(f"+RAG:     {rag_correct}/{n_samples} ({rag_acc:.1f}%)")
    print(f"Diff:     {rag_acc - baseline_acc:+.1f}%")
    
    return {
        "dataset": "MedQA",
        "baseline_accuracy": baseline_acc,
        "rag_accuracy": rag_acc,
        "improvement": rag_acc - baseline_acc,
        "samples": n_samples
    }

def benchmark_pubmedqa_comparison(n_samples: int = 30):
    """Compare Baseline vs RAG on PubMedQA"""
    print("\n" + "="*60)
    print("BENCHMARK: PubMedQA - Baseline vs RAG")
    print("="*60)
    
    with open(DATASETS_DIR / "pubmedqa_test.json", "r") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    baseline_correct = 0
    rag_correct = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        context = sample["context"][:800]
        answer = sample["final_decision"]
        
        # Baseline (with original context only)
        prompt_baseline = f"""Answer ONLY: yes, no, or maybe.

Context: {context}

Question: {question}

Answer:"""
        
        response_baseline, _ = query_ollama(prompt_baseline, max_tokens=10)
        pred_baseline = response_baseline.lower().strip()
        if "yes" in pred_baseline[:10]:
            pred_baseline = "yes"
        elif "no" in pred_baseline[:10]:
            pred_baseline = "no"
        else:
            pred_baseline = "maybe"
        
        # With additional RAG context
        rag_context = load_rag_context(question)
        if rag_context:
            prompt_rag = f"""Answer ONLY: yes, no, or maybe.

Research Context: {context}

Additional Knowledge: {rag_context[:300]}

Question: {question}

Answer:"""
        else:
            prompt_rag = prompt_baseline
        
        response_rag, _ = query_ollama(prompt_rag, max_tokens=10)
        pred_rag = response_rag.lower().strip()
        if "yes" in pred_rag[:10]:
            pred_rag = "yes"
        elif "no" in pred_rag[:10]:
            pred_rag = "no"
        else:
            pred_rag = "maybe"
        
        baseline_ok = pred_baseline == answer
        rag_ok = pred_rag == answer
        
        if baseline_ok:
            baseline_correct += 1
        if rag_ok:
            rag_correct += 1
        
        print(f"[{i+1}/{n_samples}] Base: {'✅' if baseline_ok else '❌'}{pred_baseline} | RAG: {'✅' if rag_ok else '❌'}{pred_rag} | Truth: {answer}")
    
    baseline_acc = 100 * baseline_correct / n_samples
    rag_acc = 100 * rag_correct / n_samples
    
    print(f"\n{'='*40}")
    print(f"Baseline: {baseline_correct}/{n_samples} ({baseline_acc:.1f}%)")
    print(f"+RAG:     {rag_correct}/{n_samples} ({rag_acc:.1f}%)")
    print(f"Diff:     {rag_acc - baseline_acc:+.1f}%")
    
    return {
        "dataset": "PubMedQA",
        "baseline_accuracy": baseline_acc,
        "rag_accuracy": rag_acc,
        "improvement": rag_acc - baseline_acc,
        "samples": n_samples
    }

def run_comparison():
    """Run all comparisons"""
    print("="*60)
    print("RAG COMPARISON BENCHMARK")
    print(f"Model: {MODEL}")
    print("="*60)
    
    results = []
    results.append(benchmark_medqa_comparison(n_samples=30))
    results.append(benchmark_pubmedqa_comparison(n_samples=30))
    
    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'Baseline':<12} {'+RAG':<12} {'Improvement':<12}")
    print("-"*50)
    for r in results:
        print(f"{r['dataset']:<15} {r['baseline_accuracy']:.1f}%{'':<7} {r['rag_accuracy']:.1f}%{'':<7} {r['improvement']:+.1f}%")
    
    # Save results
    output_file = DATASETS_DIR / "rag_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)
    run_comparison()
