"""
RAG-Enhanced Benchmark Evaluation Script for RAG-MedGemma
Compares baseline LLM vs RAG-enhanced performance on medical QA datasets
"""
import json
import time
import httpx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Paths - use relative paths from script location
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RESULTS_DIR = DATA_DIR / "benchmark_results"
OUTPUT_DIR = DATA_DIR / "output"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ollama config
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "gemma2:2b"

def load_rag_context(question: str, top_k: int = 5) -> str:
    """Load relevant context from GraphRAG indexed data"""
    try:
        # Load entities
        entities_df = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
        text_units_df = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
        
        # Simple keyword matching for context retrieval
        question_lower = question.lower()
        keywords = [w for w in question_lower.split() if len(w) > 3]
        
        # Find relevant entities
        relevant_entities = []
        for _, row in entities_df.iterrows():
            title = str(row.get('title', '')).lower()
            if any(kw in title for kw in keywords):
                relevant_entities.append(row.get('title', ''))
        
        # Find relevant text chunks
        relevant_chunks = []
        for _, row in text_units_df.iterrows():
            text = str(row.get('text', '')).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                relevant_chunks.append((score, row.get('text', '')[:500]))
        
        # Sort by relevance and take top_k
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c[1] for c in relevant_chunks[:top_k]]
        
        context = ""
        if relevant_entities:
            context += "RELEVANT ENTITIES:\n" + "\n".join(f"- {e}" for e in relevant_entities[:10]) + "\n\n"
        if top_chunks:
            context += "RELEVANT CONTEXT:\n" + "\n---\n".join(top_chunks)
        
        return context
    except Exception as e:
        print(f"Error loading RAG context: {e}")
        return ""

def query_ollama_with_rag(question: str, options: list = None) -> tuple[str, float]:
    """Query Ollama with RAG context"""
    start = time.time()
    
    # Get RAG context
    context = load_rag_context(question)
    
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

Based on your medical knowledge and the context above, select the correct answer.
Reply with ONLY the letter (A, B, C, or D) of the correct answer."""
    else:
        prompt = f"""You are a medical expert. Use the following medical knowledge to answer this question.

{context}

Question: {question}

Based on your medical knowledge and the context above, provide a clear answer.
Reply with ONLY one word: yes, no, or maybe."""
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
        )
    
    latency = time.time() - start
    answer = response.json().get("response", "").strip().upper()
    return answer, latency

def query_ollama_direct(question: str, options: list = None) -> tuple[str, float]:
    """Query Ollama directly (no RAG) - baseline"""
    start = time.time()
    
    if options:
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

def benchmark_medqa(limit: int = 50, use_rag: bool = True):
    """Benchmark on MedQA-USMLE dataset"""
    mode = "RAG" if use_rag else "Baseline"
    print(f"\n{'='*60}")
    print(f"Benchmarking MedQA-USMLE ({mode})")
    print("="*60)
    
    with open(DATASETS_DIR / "medqa_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:limit]
    
    results = []
    correct = 0
    total_latency = 0
    query_fn = query_ollama_with_rag if use_rag else query_ollama_direct
    
    for item in tqdm(data, desc=f"MedQA ({mode})"):
        question = item["question"]
        options = item["options"]
        answer_idx = item["answer_idx"]
        if isinstance(answer_idx, str) and answer_idx in "ABCD":
            correct_idx = ord(answer_idx) - ord('A')
        else:
            correct_idx = int(answer_idx)
        correct_letter = ["A", "B", "C", "D"][correct_idx]
        
        answer, latency = query_fn(question, options)
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
    
    print(f"\nResults ({mode}):")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(data)})")
    print(f"  Avg Latency: {avg_latency:.2f}s")
    
    return {
        "dataset": "MedQA-USMLE",
        "mode": mode,
        "samples": len(data),
        "accuracy": round(accuracy, 2),
        "avg_latency_s": round(avg_latency, 2),
        "details": results
    }

def benchmark_pubmedqa(limit: int = 50, use_rag: bool = True):
    """Benchmark on PubMedQA dataset"""
    mode = "RAG" if use_rag else "Baseline"
    print(f"\n{'='*60}")
    print(f"Benchmarking PubMedQA ({mode})")
    print("="*60)
    
    with open(DATASETS_DIR / "pubmedqa_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:limit]
    
    results = []
    correct = 0
    total_latency = 0
    query_fn = query_ollama_with_rag if use_rag else query_ollama_direct
    
    for item in tqdm(data, desc=f"PubMedQA ({mode})"):
        question = item["question"]
        expected = item["final_decision"].lower()
        
        answer, latency = query_fn(question)
        
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
    
    print(f"\nResults ({mode}):")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(data)})")
    print(f"  Avg Latency: {avg_latency:.2f}s")
    
    return {
        "dataset": "PubMedQA",
        "mode": mode,
        "samples": len(data),
        "accuracy": round(accuracy, 2),
        "avg_latency_s": round(avg_latency, 2),
        "details": results
    }

def main():
    print("="*60)
    print("RAG-MedGemma Comparison Benchmark")
    print(f"Model: {LLM_MODEL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Run RAG benchmarks
    medqa_rag = benchmark_medqa(limit=50, use_rag=True)
    pubmedqa_rag = benchmark_pubmedqa(limit=50, use_rag=True)
    
    # Save results
    results = {
        "model": LLM_MODEL,
        "timestamp": datetime.now().isoformat(),
        "rag_enabled": True,
        "benchmarks": [medqa_rag, pubmedqa_rag]
    }
    
    output_file = RESULTS_DIR / f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Dataset':<20} {'RAG Accuracy':<15} {'RAG Latency':<15}")
    print("-"*50)
    print(f"{'MedQA-USMLE':<20} {medqa_rag['accuracy']:.1f}%{'':<10} {medqa_rag['avg_latency_s']:.2f}s")
    print(f"{'PubMedQA':<20} {pubmedqa_rag['accuracy']:.1f}%{'':<10} {pubmedqa_rag['avg_latency_s']:.2f}s")
    print("="*60)
    print("\nCompare with baseline results in: data/benchmark_results/baseline_*.json")

if __name__ == "__main__":
    main()
