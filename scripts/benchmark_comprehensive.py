"""
Comprehensive Benchmark Evaluation Script for RAG-MedGemma
Compares multiple models (baseline vs RAG) on medical QA datasets
"""
import json
import time
import httpx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RESULTS_DIR = DATA_DIR / "benchmark_results"
OUTPUT_DIR = DATA_DIR / "output"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ollama config
OLLAMA_URL = "http://localhost:11434"

def load_rag_context(question: str, top_k: int = 3) -> str:
    """Load relevant context from GraphRAG indexed data using semantic matching"""
    try:
        entities_df = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
        text_units_df = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
        
        question_lower = question.lower()
        # Extract medical keywords (longer words are more meaningful)
        keywords = [w for w in question_lower.split() if len(w) > 4]
        
        # Find relevant text chunks with scoring
        relevant_chunks = []
        for _, row in text_units_df.iterrows():
            text = str(row.get('text', '')).lower()
            score = sum(2 for kw in keywords if kw in text)  # Weight keyword matches
            if score > 2:  # Only include if multiple keywords match
                relevant_chunks.append((score, row.get('text', '')[:400]))
        
        # Sort by relevance and take top_k
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c[1] for c in relevant_chunks[:top_k]]
        
        if top_chunks:
            return "MEDICAL CONTEXT:\n" + "\n---\n".join(top_chunks)
        return ""
    except Exception as e:
        return ""

def query_ollama(model: str, question: str, options: list = None, use_rag: bool = False) -> tuple[str, float]:
    """Query Ollama with configurable model and RAG"""
    start = time.time()
    
    context = load_rag_context(question) if use_rag else ""
    
    if options:
        if isinstance(options, dict):
            opts = [options.get('A', ''), options.get('B', ''), options.get('C', ''), options.get('D', '')]
        else:
            opts = options
        
        prompt = f"""You are a medical expert. Answer this USMLE-style medical question.

{context}

Question: {question}

Options:
A. {opts[0]}
B. {opts[1]}
C. {opts[2]}
D. {opts[3]}

Reply with ONLY the letter (A, B, C, or D) of the correct answer."""
    else:
        prompt = f"""You are a medical expert. Answer this medical question.

{context}

Question: {question}

Reply with ONLY one word: yes, no, or maybe."""
    
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
    
    latency = time.time() - start
    answer = response.json().get("response", "").strip().upper()
    return answer, latency

def benchmark_dataset(dataset_name: str, model: str, use_rag: bool, limit: int = 30):
    """Benchmark a model on a dataset"""
    mode = f"{model} + RAG" if use_rag else f"{model} (baseline)"
    
    if dataset_name == "medqa":
        with open(DATASETS_DIR / "medqa_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)[:limit]
        is_mcq = True
    else:
        with open(DATASETS_DIR / "pubmedqa_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)[:limit]
        is_mcq = False
    
    correct = 0
    total_latency = 0
    
    for item in tqdm(data, desc=f"{dataset_name} ({mode})"):
        if is_mcq:
            question = item["question"]
            options = item["options"]
            answer_idx = item["answer_idx"]
            if isinstance(answer_idx, str) and answer_idx in "ABCD":
                correct_idx = ord(answer_idx) - ord('A')
            else:
                correct_idx = int(answer_idx)
            correct_letter = ["A", "B", "C", "D"][correct_idx]
            
            answer, latency = query_ollama(model, question, options, use_rag)
            pred_letter = answer[0] if answer else ""
            is_correct = pred_letter == correct_letter
        else:
            question = item["question"]
            expected = item["final_decision"].lower()
            
            answer, latency = query_ollama(model, question, None, use_rag)
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
    
    return {
        "model": model,
        "dataset": dataset_name,
        "rag_enabled": use_rag,
        "samples": len(data),
        "accuracy": round(accuracy, 2),
        "avg_latency_s": round(avg_latency, 2)
    }

def main():
    print("="*70)
    print("RAG-MedGemma Comprehensive Benchmark")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    # Models to test
    models = ["medgemma"]
    datasets = ["medqa", "pubmedqa"]
    results = []
    
    for model in models:
        print(f"\n\n{'='*70}")
        print(f"Testing Model: {model}")
        print("="*70)
        
        for dataset in datasets:
            # Baseline (no RAG)
            result = benchmark_dataset(dataset, model, use_rag=False, limit=30)
            results.append(result)
            print(f"\n{dataset} ({model} baseline): {result['accuracy']:.1f}% ({result['avg_latency_s']:.2f}s)")
            
            # With RAG
            result = benchmark_dataset(dataset, model, use_rag=True, limit=30)
            results.append(result)
            print(f"{dataset} ({model} + RAG): {result['accuracy']:.1f}% ({result['avg_latency_s']:.2f}s)")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    output_file = RESULTS_DIR / f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Summary table
    print("\n\n" + "="*70)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Dataset':<12} {'Mode':<10} {'Accuracy':<12} {'Latency':<10}")
    print("-"*60)
    for r in results:
        mode = "RAG" if r['rag_enabled'] else "Baseline"
        print(f"{r['model']:<15} {r['dataset']:<12} {mode:<10} {r['accuracy']:.1f}%{'':<6} {r['avg_latency_s']:.2f}s")
    print("="*70)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
