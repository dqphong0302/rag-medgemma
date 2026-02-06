"""
Improved RAG Benchmark using proper HybridRetriever with FAISS semantic search
"""
import json
import time
import random
import httpx
import sys
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from hybrid_retriever import HybridRetriever

DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"

OLLAMA_URL = "http://localhost:11434"
MODEL = "medgemma-q8"

# Initialize retriever once
print("Initializing HybridRetriever with FAISS semantic search...")
retriever = HybridRetriever(DATA_DIR, language="en")

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

def benchmark_medqa(n_samples: int = 30):
    """Benchmark MedQA with proper Hybrid RAG"""
    print("\n" + "="*60)
    print("BENCHMARK: MedQA - Baseline vs Improved RAG")
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
        
        # === BASELINE (No RAG) ===
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
        
        # === WITH HYBRID RAG ===
        context, metadata = retriever.retrieve(
            question, 
            top_k_semantic=5,  # Increased from 3
            top_k_entities=5,  # Increased from 3
            min_score=0.2      # Lower threshold
        )
        
        if context:
            prompt_rag = f"""Use the medical knowledge below to help answer.
Answer with ONLY the letter (A-E).

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
        
        # Show retrieval quality
        status = "📊" if metadata.get('avg_semantic_score', 0) > 0.5 else "📉"
        print(f"[{i+1}/{n_samples}] Base: {'✅' if baseline_ok else '❌'}{pred_baseline} | RAG: {'✅' if rag_ok else '❌'}{pred_rag} | Truth: {answer_idx} | {status} ctx:{len(context)}")
    
    baseline_acc = 100 * baseline_correct / n_samples
    rag_acc = 100 * rag_correct / n_samples
    
    print(f"\n{'='*40}")
    print(f"MedQA Results:")
    print(f"  Baseline: {baseline_correct}/{n_samples} ({baseline_acc:.1f}%)")
    print(f"  +RAG:     {rag_correct}/{n_samples} ({rag_acc:.1f}%)")
    print(f"  Diff:     {rag_acc - baseline_acc:+.1f}%")
    
    return {
        "dataset": "MedQA",
        "baseline": baseline_acc,
        "rag": rag_acc,
        "improvement": rag_acc - baseline_acc
    }

def benchmark_pubmedqa(n_samples: int = 30):
    """Benchmark PubMedQA with proper Hybrid RAG"""
    print("\n" + "="*60)
    print("BENCHMARK: PubMedQA - Baseline vs Improved RAG")
    print("="*60)
    
    with open(DATASETS_DIR / "pubmedqa_test.json", "r") as f:
        data = json.load(f)
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    baseline_correct = 0
    rag_correct = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        pubmed_context = sample["context"][:600]  # Original context
        answer = sample["final_decision"]
        
        # === BASELINE (only PubMed context) ===
        prompt_baseline = f"""Answer ONLY: yes, no, or maybe.

Context: {pubmed_context}

Question: {question}

Answer:"""
        
        response_baseline, _ = query_ollama(prompt_baseline, max_tokens=10)
        pred_baseline = response_baseline.lower().strip()
        if "yes" in pred_baseline[:15]:
            pred_baseline = "yes"
        elif "no" in pred_baseline[:15]:
            pred_baseline = "no"
        else:
            pred_baseline = "maybe"
        
        # === WITH HYBRID RAG (additional knowledge) ===
        rag_context, metadata = retriever.retrieve(
            question,
            top_k_semantic=3,
            top_k_entities=3,
            min_score=0.25
        )
        
        if rag_context:
            prompt_rag = f"""Answer ONLY: yes, no, or maybe.

Research Context: {pubmed_context}

Additional Medical Knowledge:
{rag_context[:500]}

Question: {question}

Answer:"""
        else:
            prompt_rag = prompt_baseline
        
        response_rag, _ = query_ollama(prompt_rag, max_tokens=10)
        pred_rag = response_rag.lower().strip()
        if "yes" in pred_rag[:15]:
            pred_rag = "yes"
        elif "no" in pred_rag[:15]:
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
    print(f"PubMedQA Results:")
    print(f"  Baseline: {baseline_correct}/{n_samples} ({baseline_acc:.1f}%)")
    print(f"  +RAG:     {rag_correct}/{n_samples} ({rag_acc:.1f}%)")
    print(f"  Diff:     {rag_acc - baseline_acc:+.1f}%")
    
    return {
        "dataset": "PubMedQA",
        "baseline": baseline_acc,
        "rag": rag_acc,
        "improvement": rag_acc - baseline_acc
    }

def main():
    print("="*60)
    print("IMPROVED HYBRID RAG BENCHMARK")
    print(f"Model: {MODEL}")
    print(f"Retriever: FAISS semantic + GraphRAG entities/relationships")
    print("="*60)
    
    results = []
    results.append(benchmark_medqa(n_samples=30))
    results.append(benchmark_pubmedqa(n_samples=30))
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'Baseline':<12} {'+HybridRAG':<12} {'Improvement':<12}")
    print("-"*50)
    for r in results:
        print(f"{r['dataset']:<15} {r['baseline']:.1f}%{'':<7} {r['rag']:.1f}%{'':<7} {r['improvement']:+.1f}%")
    
    # Save
    output_file = DATASETS_DIR / "improved_rag_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)
    main()
