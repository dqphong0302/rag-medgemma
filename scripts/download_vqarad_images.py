"""
Download VQA-RAD images from HuggingFace
Dataset: flaviagiammarino/vqa-rad
"""
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
IMAGES_DIR = DATA_DIR / "datasets/vqarad_images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def download_vqarad_images():
    print("Loading VQA-RAD dataset from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
    
    print(f"Found {len(dataset)} samples in test split")
    
    # Save images
    saved_count = 0
    for i, sample in enumerate(tqdm(dataset, desc="Saving images")):
        image = sample['image']
        image_path = IMAGES_DIR / f"img_{i:04d}.png"
        
        # Skip if already exists
        if image_path.exists():
            continue
            
        # Save image
        if isinstance(image, Image.Image):
            image.save(image_path)
            saved_count += 1
    
    print(f"\nSaved {saved_count} new images to: {IMAGES_DIR}")
    print(f"Total images available: {len(list(IMAGES_DIR.glob('*.png')))}")
    
    # Also save the Q&A pairs with correct local paths
    qa_pairs = []
    for i, sample in enumerate(dataset):
        qa_pairs.append({
            "image_path": str(IMAGES_DIR / f"img_{i:04d}.png"),
            "question": sample['question'],
            "answer": sample['answer']
        })
    
    import json
    output_file = DATA_DIR / "datasets/vqarad_local.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"Saved Q&A pairs with local paths to: {output_file}")

if __name__ == "__main__":
    download_vqarad_images()
