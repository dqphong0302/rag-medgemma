"""
Hybrid RAG Benchmark: Semantic Search + GraphRAG with MedGemma
"""
import json
import time
import httpx
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hybrid_retriever import HybridRetriever

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RESULTS_DIR = DATA_DIR / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

# Initialize hybrid retriever globally
print("Initializing Hybrid Retriever...")
retriever = HybridRetriever(DATA_DIR)

def query_ollama_hybrid(model: str, question: str, options: list = None) -> tuple[str, float]:
    """Query Ollama with hybrid RAG context"""
    start = time.time()
    
    # Get hybrid context
    context, metadata = retriever.retrieve(question, top_k_semantic=3, top_k_entities=3)
    
    if options:
        if isinstance(options, dict):
            opts = [options.get('A', ''), options.get('B', ''), options.get('C', ''), options.get('D', '')]
        else:
            opts = options
        
        prompt = f"""You are a medical expert. Use the following medical knowledge to answer this USMLE question.

{context}

Question: {question}

Options:
A. {opts[0]}
B. {opts[1]}
C. {opts[2]}
D. {opts[3]}

Based on the medical context above, select the correct answer. Reply with ONLY the letter (A, B, C, or D)."""
    else:
        prompt = f"""You are a medical expert. Use the following medical knowledge to answer this question.

{context}

Question: {question}

Based on the medical context above, answer with ONLY one word: yes, no, or maybe."""
    
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
    
    latency = time.time() - start
    answer = response.json().get("response", "").strip().upper()
    return answer, latency

def benchmark_hybrid(model: str, dataset: str, limit: int = 100):
    """Benchmark model with hybrid RAG"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model} + Hybrid RAG on {dataset}")
    print("="*60)
    
    if dataset == "medqa":
        with open(DATASETS_DIR / "medqa_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)[:limit]
        is_mcq = True
    else:
        with open(DATASETS_DIR / "pubmedqa_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)[:limit]
        is_mcq = False
    
    correct = 0
    total_latency = 0
    
    for item in tqdm(data, desc=f"{dataset} ({model} hybrid)"):
        if is_mcq:
            question = item["question"]
            options = item["options"]
            answer_idx = item["answer_idx"]
            if isinstance(answer_idx, str) and answer_idx in "ABCD":
                correct_idx = ord(answer_idx) - ord('A')
            else:
                correct_idx = int(answer_idx)
            correct_letter = ["A", "B", "C", "D"][correct_idx]
            
            answer, latency = query_ollama_hybrid(model, question, options)
            pred_letter = answer[0] if answer else ""
            is_correct = pred_letter == correct_letter
        else:
            question = item["question"]
            expected = item["final_decision"].lower()
            
            answer, latency = query_ollama_hybrid(model, question, None)
            answer_lower = answer.lower()
            if "yes" in answer_lower:
                pred = "yes"
            elif "no" in answer_lower:
                pred = "no"
            else:
                pred = "maybe"
            is_correct = pred == expected
        
        if is_correct:
            correct += 1
        total_latency += latency
    
    accuracy = correct / len(data) * 100
    avg_latency = total_latency / len(data)
    
    print(f"\nResult: {accuracy:.1f}% ({correct}/{len(data)}), Avg Latency: {avg_latency:.2f}s")
    
    return {
        "model": model,
        "dataset": dataset,
        "mode": "hybrid_rag",
        "samples": len(data),
        "accuracy": round(accuracy, 2),
        "avg_latency_s": round(avg_latency, 2)
    }

def main():
    print("="*60)
    print("Hybrid RAG Benchmark (Semantic Search + GraphRAG)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    model = "medgemma-q8"
    results = []
    
    # Run hybrid benchmarks
    results.append(benchmark_hybrid(model, "medqa", limit=100))
    results.append(benchmark_hybrid(model, "pubmedqa", limit=100))
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": "hybrid_semantic_graphrag",
        "results": results
    }
    
    output_file = RESULTS_DIR / f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "="*60)
    print("HYBRID RAG BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'Accuracy':<12} {'Latency':<10}")
    print("-"*40)
    for r in results:
        print(f"{r['dataset']:<15} {r['accuracy']:.1f}%{'':<6} {r['avg_latency_s']:.2f}s")
    print("="*60)
    
    # Comparison with previous results
    print("\nCOMPARISON:")
    print("-"*50)
    print("MedGemma Results:")
    print("  MedQA:    Baseline 46.7% → Basic RAG 60.0% → Hybrid ?")
    print("  PubMedQA: Baseline 46.7% → Basic RAG 30.0% → Hybrid ?")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
