"""
Full VQA-RAD Benchmark with MedGemma Vision (via Transformers)
Tests medical image understanding on X-ray, CT, MRI images
"""
import json
import time
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
IMAGES_DIR = DATASETS_DIR / "vqarad_images"

MODEL_NAME = "google/medgemma-4b-it"

def load_model():
    """Load MedGemma Vision model"""
    print("Loading MedGemma Vision model...")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    print("Model loaded successfully!")
    return processor, model

def query_vision(processor, model, image: Image.Image, question: str) -> tuple[str, float]:
    """Query the vision model with image and question"""
    start_time = time.time()
    
    # Format for Gemma3 vision
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"You are a medical imaging expert. Answer briefly: {question}"}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
    
    response = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    gen_time = time.time() - start_time
    
    return response.strip(), gen_time

def run_vqarad_benchmark(n_samples: int = 30):
    """Run VQA-RAD benchmark"""
    print("="*60)
    print("VQA-RAD FULL BENCHMARK")
    print("="*60)
    
    # Load dataset
    with open(DATASETS_DIR / "vqarad_test.json", "r") as f:
        data = json.load(f)
    
    # Check for images
    if not IMAGES_DIR.exists():
        print(f"⚠️ Images directory not found: {IMAGES_DIR}")
        print("Please run download_vqarad_images.py first")
        return None
    
    # Filter samples with available images
    valid_samples = []
    for sample in data:
        img_name = sample.get("image", "")
        img_path = IMAGES_DIR / img_name
        if img_path.exists():
            valid_samples.append(sample)
    
    print(f"Found {len(valid_samples)} samples with images")
    
    if len(valid_samples) == 0:
        print("No valid samples found!")
        return None
    
    samples = random.sample(valid_samples, min(n_samples, len(valid_samples)))
    
    # Load model
    processor, model = load_model()
    
    correct = 0
    total_time = 0
    results = []
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"].lower().strip()
        img_name = sample["image"]
        img_path = IMAGES_DIR / img_name
        
        try:
            image = Image.open(img_path).convert("RGB")
            response, gen_time = query_vision(processor, model, image, question)
            total_time += gen_time
            
            # Simple match check
            pred = response.lower().strip()
            is_correct = answer in pred or pred in answer or pred == answer
            
            if is_correct:
                correct += 1
            
            results.append({
                "image": img_name,
                "question": question,
                "ground_truth": answer,
                "prediction": response,
                "correct": is_correct,
                "time": gen_time
            })
            
            print(f"[{i+1}/{n_samples}] {'✅' if is_correct else '❌'} Q: {question[:40]}...")
            print(f"   Pred: {response[:60]}... | Truth: {answer}")
            
        except Exception as e:
            print(f"[{i+1}/{n_samples}] ❌ Error: {e}")
    
    accuracy = 100 * correct / len(samples) if samples else 0
    avg_time = total_time / len(samples) if samples else 0
    
    print(f"\n{'='*40}")
    print(f"VQA-RAD Results: {correct}/{len(samples)} ({accuracy:.1f}%)")
    print(f"Average time: {avg_time:.1f}s")
    
    # Save results
    output = {
        "dataset": "VQA-RAD",
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "avg_time": avg_time,
        "results": results
    }
    
    output_file = DATASETS_DIR / "vqarad_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_file}")
    
    return output

if __name__ == "__main__":
    random.seed(42)
    run_vqarad_benchmark(n_samples=30)
