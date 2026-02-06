import sys
from pathlib import Path
import time
import httpx
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hybrid_retriever import HybridRetriever

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
OLLAMA_URL = "http://localhost:11434"

def query_medgemma_vi(question: str, retriever: HybridRetriever):
    print(f"\n❓ Câu hỏi: {question}")
    
    # 1. Retrieve Context (Vietnamese)
    start_time = time.time()
    context, metadata = retriever.retrieve(question, top_k_semantic=3, top_k_entities=0) # Only semantic for now
    retrieve_time = time.time() - start_time
    
    print(f"   🔍 Tìm thấy {metadata.get('semantic_chunks', 0)} đoạn văn bản liên quan ({retrieve_time:.2f}s)")
    
    # 2. Consult MedGemma
    prompt = f"""Bạn là trợ lý y tế thông minh. Dựa vào thông tin sau đây để trả lời câu hỏi của người dùng bằng Tiếng Việt.
Nếu thông tin không có trong văn bản, hãy dùng kiến thức của bạn nhưng cảnh báo người dùng.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""
    
    start_gen = time.time()
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "medgemma-q8", 
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 4096
                }
            }
        )
    gen_time = time.time() - start_gen
    
    answer = response.json().get("response", "").strip()
    print(f"   🤖 MedGemma trả lời ({gen_time:.2f}s):")
    print(f"   {'-'*40}")
    print(f"   {answer}")
    print(f"   {'-'*40}")

def main():
    print("="*60)
    print("DEMO: RAG-MedGemma Tiếng Việt (Vietnamese Medical RAG)")
    print(f"Dataset: hungnm/vietnamese-medical-qa")
    print(f"Model: MedGemma-4B-Q8 + Multilingual Embedding")
    print("="*60)
    
    # Initialize Retriever in Vietnamese mode
    retriever = HybridRetriever(DATA_DIR, language="vi")
    
    # Sample Questions
    questions = [
        "Triệu chứng của sốt xuất huyết là gì?",
        "Bị đau đầu, chóng mặt và buồn nôn là dấu hiệu bệnh gì?",
        "Trẻ sơ sinh bị vàng da có nguy hiểm không?",
        "Làm sao để phòng ngừa bệnh tiểu đường?",
        "Đau bụng dưới bên phải âm ỉ là bị gì?"
    ]
    
    for q in questions:
        query_medgemma_vi(q, retriever)

if __name__ == "__main__":
    main()
