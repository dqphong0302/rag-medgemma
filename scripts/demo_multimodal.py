"""
MedGemma Vision Multimodal Demo
Processes medical images (X-ray, CT, MRI) with question answering
"""
import json
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
VQARAD_FILE = DATA_DIR / "datasets/vqarad_local.json"

# Model configuration
MODEL_ID = "google/medgemma-4b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load MedGemma Vision model"""
    print(f"Loading MedGemma Vision on {DEVICE}...")
    print("This may take a few minutes on first run (downloading ~8GB)...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    return processor, model

def process_medical_image(processor, model, image_path: str, question: str):
    """Process a medical image with a question"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create prompt with image token for Gemma3 Vision
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"You are an expert radiologist. Analyze the medical image and answer the question concisely.\nQuestion: {question}\nAnswer:"}
            ]
        }
    ]
    
    # Process inputs using chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    # Generate response
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    
    gen_time = time.time() - start_time
    
    # Decode response - skip input tokens
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    return response.strip(), gen_time

def run_demo():
    """Run multimodal demo on VQA-RAD samples"""
    print("="*60)
    print("MedGemma Vision Multimodal Demo")
    print("Dataset: VQA-RAD (Medical Visual Question Answering)")
    print("="*60)
    
    # Load Q&A pairs
    with open(VQARAD_FILE, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Load model
    processor, model = load_model()
    
    # Select diverse samples (different modalities)
    sample_indices = [0, 10, 25, 50, 80]  # Mix of X-ray, CT, MRI
    
    results = []
    for idx in sample_indices:
        if idx >= len(qa_pairs):
            continue
            
        sample = qa_pairs[idx]
        image_path = sample['image_path']
        question = sample['question']
        ground_truth = sample['answer']
        
        # Skip if image doesn't exist
        if not Path(image_path).exists():
            print(f"⚠️ Image not found: {image_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📷 Image: {Path(image_path).name}")
        print(f"❓ Question: {question}")
        print(f"✅ Ground Truth: {ground_truth}")
        
        # Get model prediction
        prediction, gen_time = process_medical_image(processor, model, image_path, question)
        
        print(f"🤖 MedGemma: {prediction}")
        print(f"⏱️ Time: {gen_time:.2f}s")
        
        # Check if correct (simple match)
        is_correct = ground_truth.lower() in prediction.lower() or prediction.lower() in ground_truth.lower()
        print(f"{'✅ CORRECT' if is_correct else '❌ DIFFERENT'}")
        
        results.append({
            "image": Path(image_path).name,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": is_correct,
            "time": gen_time
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    correct = sum(1 for r in results if r['correct'])
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"Avg Time: {sum(r['time'] for r in results)/len(results):.2f}s")
    
    # Save results
    output_file = DATA_DIR / "datasets/vqarad_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    run_demo()
