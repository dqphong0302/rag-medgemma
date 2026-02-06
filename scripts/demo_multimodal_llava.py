"""
Multimodal Medical VQA Demo using LLaVA via Ollama
Processes medical images (X-ray, CT, MRI) with question answering
"""
import json
import time
import base64
import httpx
from pathlib import Path
from PIL import Image
import io

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
VQARAD_FILE = DATA_DIR / "datasets/vqarad_local.json"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
MODEL = "llava:7b"

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 for Ollama"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_llava(image_path: str, question: str) -> tuple[str, float]:
    """Query LLaVA model with image and question"""
    # Prepare prompt
    prompt = f"""You are an expert radiologist analyzing a medical image.
Look at this medical image carefully and answer the question.
Be concise and specific.

Question: {question}
Answer:"""
    
    # Convert image to base64
    image_b64 = image_to_base64(image_path)
    
    # Make request
    start_time = time.time()
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 2048
                }
            }
        )
    gen_time = time.time() - start_time
    
    answer = response.json().get("response", "").strip()
    return answer, gen_time

def run_demo():
    """Run multimodal demo on VQA-RAD samples"""
    print("="*60)
    print("Multimodal Medical VQA Demo (LLaVA)")
    print("Dataset: VQA-RAD (X-ray, CT, MRI)")
    print("="*60)
    
    # Load Q&A pairs
    with open(VQARAD_FILE, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Select diverse samples
    sample_indices = [0, 15, 30, 55, 80]
    
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
        try:
            prediction, gen_time = query_llava(image_path, question)
            
            print(f"🤖 LLaVA: {prediction[:200]}...")
            print(f"⏱️ Time: {gen_time:.2f}s")
            
            # Check if correct (simple match)
            is_correct = ground_truth.lower() in prediction.lower() or prediction.lower() in ground_truth.lower()
            print(f"{'✅ MATCH' if is_correct else '❌ DIFFERENT'}")
            
            results.append({
                "image": Path(image_path).name,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": is_correct,
                "time": gen_time
            })
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("="*60)
        correct = sum(1 for r in results if r['correct'])
        print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
        print(f"Avg Time: {sum(r['time'] for r in results)/len(results):.2f}s")
        
        # Save results
        output_file = DATA_DIR / "datasets/vqarad_llava_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    run_demo()
