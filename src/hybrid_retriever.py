"""
Hybrid RAG System: Semantic Search + GraphRAG
Combines vector similarity with entity/relationship context for improved retrieval
"""
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

class HybridRetriever:
    """Hybrid retrieval combining semantic search with GraphRAG knowledge graph"""
    
    def __init__(self, data_dir: Path, embedding_model: str = "all-MiniLM-L6-v2", language: str = "en"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "output"
        self.language = language
        
        # Select model based on language
        if language == "vi":
            embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            print(f"Using Vietnamese mode with model: {embedding_model}")
        
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(embedding_model)
        
        if language == "en":
            # Load GraphRAG data (English default)
            print("Loading GraphRAG data...")
            self.text_units = pd.read_parquet(self.output_dir / "text_units.parquet")
            self.entities = pd.read_parquet(self.output_dir / "entities.parquet")
            self.relationships = pd.read_parquet(self.output_dir / "relationships.parquet")
            self.index_path = self.output_dir / "semantic_index.faiss"
            self.embeddings_path = self.output_dir / "chunk_embeddings.npy"
        else:
            # Load Vietnamese data directly from JSON
            print("Loading Vietnamese data...")
            vi_data_path = self.data_dir / "datasets/vi_medqa.json"
            with open(vi_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Create DataFrame with 'text' column for compatibility
            # Format: "Question: ... \nAnswer: ..."
            texts = [f"Hỏi: {item['question']}\nĐáp: {item['answer']}" for item in data]
            self.text_units = pd.DataFrame({'text': texts})
            self.entities = pd.DataFrame(columns=['title', 'type', 'description']) # Empty for now
            self.relationships = pd.DataFrame(columns=['source', 'target', 'description']) # Empty for now
            self.index_path = self.data_dir / "datasets/vi_semantic_index.faiss"
            self.embeddings_path = self.data_dir / "datasets/vi_embeddings.npy"
        
        # Build or load vector index
        if self.index_path.exists() and self.embeddings_path.exists():
            print("Loading existing vector index...")
            self.index = faiss.read_index(str(self.index_path))
            self.embeddings = np.load(self.embeddings_path)
        else:
            print("Building vector index (this may take a moment)...")
            self._build_index()
        
        print(f"Hybrid retriever ready: {len(self.text_units)} chunks")
    
    def _build_index(self):
        """Build FAISS index from text chunks"""
        texts = self.text_units['text'].tolist()
        
        # Generate embeddings
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        # Save index
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.embeddings_path, self.embeddings)
        print(f"Index saved: {len(texts)} chunks indexed")
    
    def _get_entity_context(self, query: str, top_k: int = 5) -> str:
        """Get relevant entities from GraphRAG"""
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) > 3]
        
        # Find matching entities
        matching_entities = []
        for _, row in self.entities.iterrows():
            title = str(row.get('title', '')).lower()
            desc = str(row.get('description', '')).lower() if pd.notna(row.get('description')) else ''
            
            score = sum(2 for kw in keywords if kw in title)
            score += sum(1 for kw in keywords if kw in desc)
            
            if score > 0:
                matching_entities.append({
                    'title': row.get('title', ''),
                    'type': row.get('type', ''),
                    'description': row.get('description', ''),
                    'score': score
                })
        
        # Sort by score and take top_k
        matching_entities.sort(key=lambda x: x['score'], reverse=True)
        top_entities = matching_entities[:top_k]
        
        if not top_entities:
            return ""
        
        context = "MEDICAL ENTITIES:\n"
        for e in top_entities:
            context += f"• {e['title']} ({e['type']})"
            if e['description'] and pd.notna(e['description']):
                context += f": {e['description'][:150]}"
            context += "\n"
        
        return context
    
    def _get_relationships(self, entity_titles: List[str], top_k: int = 3) -> str:
        """Get relationships for given entities"""
        if not entity_titles:
            return ""
        
        related = []
        for _, row in self.relationships.iterrows():
            source = str(row.get('source', '')).lower()
            target = str(row.get('target', '')).lower()
            
            for title in entity_titles:
                if title.lower() in source or title.lower() in target:
                    related.append({
                        'source': row.get('source', ''),
                        'target': row.get('target', ''),
                        'description': row.get('description', '')
                    })
                    break
        
        if not related:
            return ""
        
        context = "\nRELATIONSHIPS:\n"
        for r in related[:top_k]:
            desc = r['description'][:100] if r['description'] and pd.notna(r['description']) else ''
            context += f"• {r['source']} → {r['target']}: {desc}\n"
        
        return context
    
    def retrieve(self, query: str, top_k_semantic: int = 3, top_k_entities: int = 3, 
                 min_score: float = 0.3) -> Tuple[str, dict]:
        """
        Hybrid retrieval combining semantic search + GraphRAG
        
        Returns:
            context: Combined context string
            metadata: Dict with retrieval details
        """
        # 1. Semantic search
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k_semantic * 2)  # Get more, filter by score
        
        semantic_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score:
                text = self.text_units.iloc[idx]['text']
                semantic_chunks.append({
                    'text': text[:400],
                    'score': float(score)
                })
        
        semantic_chunks = semantic_chunks[:top_k_semantic]
        
        # 2. GraphRAG entity context
        entity_context = self._get_entity_context(query, top_k_entities)
        
        # 3. Get entity titles for relationship lookup
        entity_titles = []
        for _, row in self.entities.iterrows():
            title = str(row.get('title', '')).lower()
            if any(kw in title for kw in query.lower().split() if len(kw) > 3):
                entity_titles.append(row.get('title', ''))
        
        # 4. Get relationships
        relationship_context = self._get_relationships(entity_titles[:3], top_k=3)
        
        # 5. Combine contexts
        context_parts = []
        
        if entity_context:
            context_parts.append(entity_context)
        
        if relationship_context:
            context_parts.append(relationship_context)
        
        if semantic_chunks:
            context_parts.append("\nRELEVANT MEDICAL TEXT:")
            for i, chunk in enumerate(semantic_chunks, 1):
                context_parts.append(f"\n[{i}] (relevance: {chunk['score']:.2f})\n{chunk['text']}")
        
        context = "\n".join(context_parts) if context_parts else ""
        
        metadata = {
            'semantic_chunks': len(semantic_chunks),
            'entity_context': bool(entity_context),
            'relationships': bool(relationship_context),
            'avg_semantic_score': np.mean([c['score'] for c in semantic_chunks]) if semantic_chunks else 0
        }
        
        return context, metadata


# Convenience function
def get_hybrid_context(query: str, data_dir: str = None) -> str:
    """Quick function to get hybrid RAG context for a query"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    retriever = HybridRetriever(data_dir)
    context, _ = retriever.retrieve(query)
    return context


if __name__ == "__main__":
    # Test the hybrid retriever
    data_dir = Path(__file__).parent.parent / "data"
    retriever = HybridRetriever(data_dir)
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes heart failure?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        context, metadata = retriever.retrieve(query)
        print(f"Metadata: {metadata}")
        print(f"Context length: {len(context)} chars")
        print(f"Context preview:\n{context[:500]}...")
