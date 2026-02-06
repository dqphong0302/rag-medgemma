"""
Clinical Decision Support API
FastAPI backend using Ollama directly with GraphRAG indexed data
"""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import httpx
import json

app = FastAPI(title="MedGemma Clinical Decision Support")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths - use relative paths from script location
SRC_DIR = Path(__file__).parent.resolve()
DATA_DIR = SRC_DIR.parent / "data"
OUTPUT_DIR = DATA_DIR / "output"
STATIC_DIR = SRC_DIR / "static"

# Ollama config
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "medgemma-q8"  # Updated to MedGemma for better medical accuracy

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# Load indexed data
def load_context():
    """Load entities and relationships from GraphRAG output"""
    entities_df = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
    relationships_df = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")
    text_units_df = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
    
    # Build context from entities
    entities_text = []
    for _, row in entities_df.iterrows():
        name = row.get('name', row.get('title', 'Unknown'))
        desc = row.get('description', '')
        if desc:
            entities_text.append(f"- {name}: {desc[:200]}")
    
    # Build context from text units
    text_chunks = []
    for _, row in text_units_df.iterrows():
        text = row.get('text', '')
        if text:
            text_chunks.append(text[:300])
    
    return {
        "entities": entities_text[:20],
        "text_chunks": text_chunks[:5]
    }

async def query_ollama(prompt: str) -> str:
    """Query Ollama with the given prompt"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")
        return response.json().get("response", "")

@app.get("/")
async def root():
    """Serve the main UI"""
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge base using Ollama with GraphRAG context"""
    try:
        # Load context
        context = load_context()
        
        # Build prompt with context
        prompt = f"""You are a clinical decision support assistant. Use the following medical knowledge to answer the question.

KNOWN ENTITIES:
{chr(10).join(context['entities'])}

RELEVANT TEXT:
{chr(10).join(context['text_chunks'])}

QUESTION: {request.question}

Provide a helpful, accurate response based on the available information. If the information is not available, say so clearly.

RESPONSE:"""
        
        # Query Ollama
        answer = await query_ollama(prompt)
        
        return QueryResponse(
            answer=answer.strip(),
            sources=context['entities'][:3] if context['entities'] else ["Sample patient record"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MedGemma CDS"}

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
