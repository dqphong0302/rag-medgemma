"""
Create Pure Medical Knowledge Base from MedQA
Extract only the answer explanations and key medical facts, removing the Q&A structure
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
    print("CREATING PURE MEDICAL KNOWLEDGE BASE")
    print("="*60)
    
    # Load MedQA training dataset
    print("\n[1/4] Loading MedQA training data...")
    ds = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
    print(f"Loaded {len(ds)} samples")
    
    # Extract pure knowledge - focus on answers and medical concepts
    print("\n[2/4] Extracting pure medical knowledge...")
    knowledge_docs = []
    
    for item in ds:
        # Extract the correct answer as a medical fact
        answer = item['answer']
        answer_idx = item['answer_idx']
        question = item['question']
        metamap = item['metamap_phrases']
        
        # Get the option that was correct
        correct_option_text = item['options'].get(answer_idx, answer)
        
        # Create a factual statement - not a question format
        # Focus on what the answer IS, not asking about it
        key_concepts = ', '.join(metamap[:8]) if metamap else ''
        
        # Create medical fact document
        fact_doc = f"""Medical Fact: {answer}

Clinical Context: {correct_option_text}

Related Concepts: {key_concepts}"""
        
        knowledge_docs.append(fact_doc)
        
        # Also add metamap phrases as separate concepts
        if len(metamap) > 5:
            concepts_doc = f"USMLE Concept: {', '.join(metamap[:15])}"
            knowledge_docs.append(concepts_doc)
    
    print(f"Created {len(knowledge_docs)} knowledge documents")
    
    # Load embedding model
    print("\n[3/4] Generating embeddings...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    batch_size = 512
    all_embeddings = []
    
    for i in range(0, len(knowledge_docs), batch_size):
        batch = knowledge_docs[i:i+batch_size]
        embeddings = encoder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)
        if (i + batch_size) % 5000 == 0:
            print(f"  Processed {min(i+batch_size, len(knowledge_docs))}/{len(knowledge_docs)}")
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("\n[4/4] Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save
    new_index_path = OUTPUT_DIR / "medical_knowledge_index.faiss"
    new_embeddings_path = OUTPUT_DIR / "medical_knowledge_embeddings.npy"
    new_docs_path = OUTPUT_DIR / "medical_knowledge_docs.json"
    
    faiss.write_index(index, str(new_index_path))
    np.save(new_embeddings_path, embeddings)
    with open(new_docs_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_docs, f, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("KNOWLEDGE BASE CREATED")
    print("="*60)
    print(f"Documents: {len(knowledge_docs)}")
    print(f"Index size: {new_index_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Sample: {knowledge_docs[0]}")

if __name__ == "__main__":
    main()
