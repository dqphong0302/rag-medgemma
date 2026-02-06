"""
Index MedQA Training Data into RAG System
Adds 10,178 USMLE Q&A pairs with explanations to FAISS index
"""
import json
import numpy as np
import faiss
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "output"

def main():
    print("="*60)
    print("INDEXING MedQA TRAINING DATA INTO RAG")
    print("="*60)
    
    # Load MedQA training dataset
    print("\n[1/4] Loading MedQA training dataset...")
    ds = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
    print(f"Loaded {len(ds)} samples")
    
    # Prepare documents for indexing
    print("\n[2/4] Preparing documents...")
    documents = []
    for item in ds:
        # Create rich document with Q, options, and answer
        options_text = "\n".join([f"  {k}: {v}" for k, v in item['options'].items()])
        
        doc = f"""USMLE Question:
{item['question']}

Options:
{options_text}

Correct Answer: {item['answer_idx']} - {item['answer']}

Key Medical Concepts: {', '.join(item['metamap_phrases'][:10])}"""
        
        documents.append(doc)
    
    print(f"Created {len(documents)} documents")
    print(f"Sample document:\n{documents[0][:500]}...")
    
    # Load embedding model
    print("\n[3/4] Loading embedding model and generating embeddings...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings in batches
    batch_size = 256
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embeddings = encoder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)
        print(f"  Processed {min(i+batch_size, len(documents))}/{len(documents)}")
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("\n[4/4] Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save new index
    new_index_path = OUTPUT_DIR / "medqa_training_index.faiss"
    new_embeddings_path = OUTPUT_DIR / "medqa_training_embeddings.npy"
    new_docs_path = OUTPUT_DIR / "medqa_training_docs.json"
    
    faiss.write_index(index, str(new_index_path))
    np.save(new_embeddings_path, embeddings)
    with open(new_docs_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"\nIndex saved: {new_index_path}")
    print(f"Embeddings saved: {new_embeddings_path}")
    print(f"Documents saved: {new_docs_path}")
    
    # Statistics
    print("\n" + "="*60)
    print("INDEXING COMPLETE")
    print("="*60)
    print(f"Total documents indexed: {len(documents)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Index size: {new_index_path.stat().st_size / 1024 / 1024:.2f} MB")
    
if __name__ == "__main__":
    main()
