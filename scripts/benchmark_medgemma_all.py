"""
Comprehensive MedGemma Benchmark
Runs all benchmarks: MedQA, PubMedQA, VQA-RAD, Vietnamese Medical QA
Uses MedGemma 4B-IT via HuggingFace Transformers
"""
import json
import time
import random
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"

# Model configuration
MODEL_ID = "google/medgemma-4b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache
_model = None
_processor = None
_tokenizer = None

def load_text_model():
    """Load MedGemma for text-only tasks"""
    global _model, _tokenizer
    if _model is None:
        print(f"Loading MedGemma Text Model on {DEVICE}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
            device_map="auto"
        )
        print("Text model loaded!")
    return _tokenizer, _model

def load_vision_model():
    """Load MedGemma for vision tasks"""
    global _model, _processor
    if _processor is None:
        print(f"Loading MedGemma Vision Model on {DEVICE}...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
            device_map="auto"
        )
        print("Vision model loaded!")
    return _processor, _model

def query_text_model(tokenizer, model, prompt: str, max_tokens: int = 128) -> tuple[str, float]:
    """Query MedGemma text model"""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    gen_time = time.time() - start_time
    
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip(), gen_time

def query_vision_model(processor, model, image_path: str, question: str) -> tuple[str, float]:
    """Query MedGemma vision model"""
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"You are an expert radiologist. Answer concisely.\nQuestion: {question}\nAnswer:"}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen_time = time.time() - start_time
    
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip(), gen_time

def benchmark_medqa(tokenizer, model, n_samples: int = 30):
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
        answer = sample["answer"]
        
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        prompt = f"""You are a medical expert. Answer this USMLE question.
Question: {question}
Options:
{options_text}

Reply with ONLY the letter (A, B, C, D, or E) of the correct answer."""
        
        response, gen_time = query_text_model(tokenizer, model, prompt)
        total_time += gen_time
        
        # Extract answer letter
        pred = response.strip().upper()[:1]
        is_correct = pred == answer
        if is_correct:
            correct += 1
        
        print(f"[{i+1}/{n_samples}] {'✅' if is_correct else '❌'} Pred: {pred}, Truth: {answer} ({gen_time:.1f}s)")
    
    accuracy = 100 * correct / n_samples
    avg_time = total_time / n_samples
    print(f"\nMedQA Results: {correct}/{n_samples} ({accuracy:.1f}%) | Avg: {avg_time:.1f}s")
    return {"dataset": "MedQA", "accuracy": accuracy, "correct": correct, "total": n_samples, "avg_time": avg_time}

def benchmark_pubmedqa(tokenizer, model, n_samples: int = 30):
    """Benchmark on PubMedQA"""
    print("\n" + "="*60)
    print("BENCHMARK: PubMedQA")
    print("="*60)
    
    with open(DATASETS_DIR / "pubmedqa_test.json", "r") as f:
        data = json.load(f)
    
    samples = random.sample(list(data.values()), min(n_samples, len(data)))
    correct = 0
    total_time = 0
    
    for i, sample in enumerate(samples):
        question = sample["QUESTION"]
        context = " ".join(sample["CONTEXTS"])[:1500]  # Truncate context
        answer = sample["final_decision"]
        
        prompt = f"""Based on the research context, answer the question with 'yes', 'no', or 'maybe'.

Context: {context}

Question: {question}

Answer (yes/no/maybe):"""
        
        response, gen_time = query_text_model(tokenizer, model, prompt)
        total_time += gen_time
        
        pred = response.lower().strip()
        if "yes" in pred:
            pred = "yes"
        elif "no" in pred:
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

def benchmark_vietnamese(tokenizer, model, n_samples: int = 30):
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
        reference = sample["answer"][:200]  # Truncate reference
        
        prompt = f"""Bạn là bác sĩ chuyên khoa. Trả lời ngắn gọn câu hỏi y tế sau bằng tiếng Việt.

Câu hỏi: {question}

Trả lời:"""
        
        response, gen_time = query_text_model(tokenizer, model, prompt, max_tokens=200)
        total_time += gen_time
        
        # For Vietnamese, we do qualitative evaluation (no exact match)
        results.append({
            "question": question[:100],
            "reference": reference,
            "prediction": response[:200],
            "time": gen_time
        })
        
        print(f"[{i+1}/{n_samples}] Q: {question[:50]}... ({gen_time:.1f}s)")
        print(f"   A: {response[:100]}...")
    
    avg_time = total_time / n_samples
    print(f"\nVietnamese QA: {n_samples} samples | Avg: {avg_time:.1f}s")
    return {"dataset": "Vietnamese", "samples": n_samples, "avg_time": avg_time, "results": results}

def benchmark_vqarad(processor, model, n_samples: int = 20):
    """Benchmark on VQA-RAD (Medical Image QA)"""
    print("\n" + "="*60)
    print("BENCHMARK: VQA-RAD (Medical Images)")
    print("="*60)
    
    with open(DATASETS_DIR / "vqarad_local.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter samples with existing images
    valid_samples = [s for s in data if Path(s["image_path"]).exists()]
    samples = random.sample(valid_samples, min(n_samples, len(valid_samples)))
    
    correct = 0
    total_time = 0
    
    for i, sample in enumerate(samples):
        image_path = sample["image_path"]
        question = sample["question"]
        answer = sample["answer"].lower()
        
        response, gen_time = query_vision_model(processor, model, image_path, question)
        total_time += gen_time
        
        pred = response.lower()
        is_correct = answer in pred or pred in answer
        if is_correct:
            correct += 1
        
        print(f"[{i+1}/{n_samples}] {'✅' if is_correct else '❌'} Q: {question[:40]}... ({gen_time:.1f}s)")
    
    accuracy = 100 * correct / n_samples
    avg_time = total_time / n_samples
    print(f"\nVQA-RAD Results: {correct}/{n_samples} ({accuracy:.1f}%) | Avg: {avg_time:.1f}s")
    return {"dataset": "VQA-RAD", "accuracy": accuracy, "correct": correct, "total": n_samples, "avg_time": avg_time}

def run_all_benchmarks():
    """Run all benchmarks"""
    print("="*60)
    print("COMPREHENSIVE MedGemma BENCHMARK")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    results = []
    
    # Text benchmarks
    tokenizer, model = load_text_model()
    results.append(benchmark_medqa(tokenizer, model, n_samples=30))
    results.append(benchmark_pubmedqa(tokenizer, model, n_samples=30))
    results.append(benchmark_vietnamese(tokenizer, model, n_samples=20))
    
    # Clear text model from memory
    del model, tokenizer
    _model = None
    _tokenizer = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Vision benchmark
    processor, model = load_vision_model()
    results.append(benchmark_vqarad(processor, model, n_samples=20))
    
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
    output_file = DATASETS_DIR / "medgemma_comprehensive_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    run_all_benchmarks()
