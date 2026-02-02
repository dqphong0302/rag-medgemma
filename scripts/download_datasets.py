"""
Download Medical QA Datasets from HuggingFace
Includes text-based QA and image-based VQA datasets
"""
from datasets import load_dataset
import json
from pathlib import Path

OUTPUT_DIR = Path("d:/medgemma/data/datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_medqa():
    """Download MedQA-USMLE dataset"""
    print("Downloading MedQA-USMLE...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    
    test_data = []
    for item in dataset['test']:
        test_data.append({
            "question": item['question'],
            "options": item['options'],
            "answer_idx": item['answer_idx'],
            "answer": item['options'][item['answer_idx']]
        })
    
    with open(OUTPUT_DIR / "medqa_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(test_data)} MedQA questions")
    return len(test_data)

def download_pubmedqa():
    """Download PubMedQA dataset (labeled subset)"""
    print("Downloading PubMedQA...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    
    test_data = []
    for item in dataset['train']:
        test_data.append({
            "question": item['question'],
            "context": " ".join(item['context']['contexts']) if 'context' in item else "",
            "long_answer": item.get('long_answer', ''),
            "final_decision": item['final_decision']
        })
    
    with open(OUTPUT_DIR / "pubmedqa_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(test_data)} PubMedQA questions")
    return len(test_data)

def download_vqarad():
    """Download VQA-RAD radiology image VQA dataset"""
    print("Downloading VQA-RAD (radiology images)...")
    try:
        dataset = load_dataset("flaviagiammarino/vqa-rad")
        
        test_data = []
        images_dir = OUTPUT_DIR / "vqarad_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for i, item in enumerate(dataset['test']):
            img = item['image']
            img_path = images_dir / f"img_{i:04d}.png"
            img.save(img_path)
            
            test_data.append({
                "image_path": str(img_path),
                "question": item['question'],
                "answer": item['answer']
            })
        
        with open(OUTPUT_DIR / "vqarad_test.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(test_data)} VQA-RAD samples")
        return len(test_data)
    except Exception as e:
        print(f"VQA-RAD download failed: {e}")
        return 0

def download_pathvqa():
    """Download PathVQA pathology image VQA dataset"""
    print("Downloading PathVQA (pathology images)...")
    try:
        dataset = load_dataset("flaviagiammarino/path-vqa")
        
        test_data = []
        images_dir = OUTPUT_DIR / "pathvqa_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit to 500 for disk space
        for i, item in enumerate(dataset['test']):
            if i >= 500:
                break
            img = item['image']
            img_path = images_dir / f"img_{i:04d}.png"
            img.save(img_path)
            
            test_data.append({
                "image_path": str(img_path),
                "question": item['question'],
                "answer": item['answer']
            })
        
        with open(OUTPUT_DIR / "pathvqa_test.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(test_data)} PathVQA samples")
        return len(test_data)
    except Exception as e:
        print(f"PathVQA download failed: {e}")
        return 0

def create_rag_corpus():
    """Create corpus from PubMedQA contexts for RAG indexing"""
    print("Creating RAG corpus from PubMedQA...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    
    corpus_dir = Path("d:/medgemma/data/input")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    for f in corpus_dir.glob("*.txt"):
        f.unlink()
    
    doc_count = 0
    for i, item in enumerate(dataset['train']):
        if 'context' in item and item['context']['contexts']:
            context_text = " ".join(item['context']['contexts'])
            with open(corpus_dir / f"pubmed_{i:04d}.txt", "w", encoding="utf-8") as f:
                f.write(f"Question: {item['question']}\n\n")
                f.write(f"Context: {context_text}\n\n")
                f.write(f"Answer: {item.get('long_answer', item['final_decision'])}\n")
            doc_count += 1
    
    print(f"Created {doc_count} documents for RAG")
    return doc_count

if __name__ == "__main__":
    print("=" * 60)
    print("Medical Dataset Downloader for RAG-MedGemma")
    print("=" * 60)
    
    # Text-based datasets
    print("\n--- Text QA Datasets ---")
    medqa_count = download_medqa()
    pubmedqa_count = download_pubmedqa()
    corpus_count = create_rag_corpus()
    
    # Image-based datasets
    print("\n--- Image VQA Datasets ---")
    vqarad_count = download_vqarad()
    pathvqa_count = download_pathvqa()
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  MedQA (USMLE):       {medqa_count:>6} questions")
    print(f"  PubMedQA:            {pubmedqa_count:>6} questions")
    print(f"  RAG Corpus:          {corpus_count:>6} documents")
    print(f"  VQA-RAD (radiology): {vqarad_count:>6} image-question pairs")
    print(f"  PathVQA (pathology): {pathvqa_count:>6} image-question pairs")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'graphrag index --root d:\\medgemma\\data' to index the corpus")
    print("2. Run 'python scripts/benchmark.py' to evaluate the model")
