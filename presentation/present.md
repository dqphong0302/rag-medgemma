# KỊCH BẢN TRÌNH BÀY (PRESENTATION SCRIPT)

**Đề tài:** Phát triển mô hình RAG-MedGemma kết hợp EdgeAI cho hệ thống hỗ trợ ra quyết định lâm sàng

---

## 👋 Mở đầu & Giới thiệu (Slide 1-2)

### Slide 1: Title Slide

**"Xin chào quý thầy cô và các bạn đồng nghiệp. Tên tôi là Phong Đăng. Hôm nay tôi xin trình bày đề tài: 'Phát triển mô hình RAG-MedGemma kết hợp EdgeAI cho hệ thống hỗ trợ ra quyết định lâm sàng'."**

* **Nhấn mạnh:** Đây là giải pháp CDSS (Clinical Decision Support System) chạy hoàn toàn **offline** trên thiết bị cá nhân (Edge), đảm bảo bảo mật tuyệt đối.

### Slide 2: Nội dung trình bày

**"Bài báo cáo gồm 6 phần chính, đi từ việc đặt vấn đề thực tế, đề xuất giải pháp kỹ thuật, đến các thực nghiệm và kết quả quan trọng mà chúng tôi đã đạt được."**

---

## 🎯 Đặt vấn đề & Câu hỏi nghiên cứu (Slide 3-4)

### Slide 3: Đặt vấn đề

**"Tại sao chúng ta cần AI chạy tại chỗ (Edge AI) thay vì dùng ChatGPT hay Gemini trên Cloud?"**

* **Bảo mật:** Dữ liệu bệnh án là tối mật, không được phép gửi lên server nước ngoài.
* **Kết nối:** Các bệnh viện vùng sâu vùng xa mạng internet không ổn định.
* **Chi phí:** API Cloud rất đắt đỏ khi triển khai diện rộng.
* **Hallucination:** Các mô hình LLM thông thường hay 'bịa' thông tin, rất nguy hiểm trong y tế.

### Slide 4: Câu hỏi nghiên cứu (RQs)

**"Từ đó, đề tài tập trung giải quyết 3 câu hỏi lớn:"**

1. **RQ1:** Liệu máy tính cá nhân (RAM 16GB) có chạy nổi LLM y tế chuyên sâu không?
2. **RQ2:** Làm sao để RAG (Retrieval Augmented Generation) thực sự hiệu quả? Format dữ liệu nào là tốt nhất?
3. **RQ3:** Đánh đổi giữa Edge và Cloud là gì? Hiệu năng thua kém bao nhiêu?

---

## 💡 Giải pháp & Kiến trúc (Slide 5-8)

### Slide 5: Giải pháp tổng thể

**"Chúng tôi đề xuất mô hình gồm 3 thành phần:"**

* **MedGemma-Q8:** "Bộ não" AI chuyên y khoa (đã được lượng hoá để nhẹ hơn).
* **Pure Knowledge RAG:** "Thư viện" kiến thức y học chuẩn xác để tra cứu.
* **Ollama:** Nền tảng chạy AI offline trên biên (Edge).

### Slide 6: Model MedGemma

**"Về model, chúng tôi chọn Google MedGemma (4 tỷ tham số)."**

* **Điểm đặc biệt:** Chúng tôi sử dụng bản **Quantized Q8 (8-bit)**.
* **Tác dụng:** Giảm dung lượng xuống chỉ còn **4.13 GB**, chạy mượt mà trên laptop thường mà vẫn giữ được 98% độ chính xác gốc.

### Slide 7: Kiến trúc RAG

**"Đây là luồng xử lý của hệ thống:"**

1. Câu hỏi bác sĩ -> Hệ thống tìm kiếm (Retriever).
2. Truy xuất kiến thức liên quan từ kho dữ liệu (FAISS).
3. Gộp kiến thức chuẩn + câu hỏi -> Gửi cho AI trả lời.
**=> AI không trả lời 'chay', mà trả lời dựa trên sách vở.**

### Slide 8: Thiết kế Knowledge Base (Quan trọng)

**"Chúng tôi đã thử nghiệm 3 cách tổ chức dữ liệu:"**

1. **GraphRAG:** Dạng đồ thị các thực thể.
2. **Q&A Format:** Dạng câu hỏi - đáp án (ví dụ từ đề thi MedQA).
3. **Pure Facts:** Dạng kiến thức thuần túy (các sự thật y khoa).

---

## 🧪 Thực nghiệm & Kết quả (Slide 9-11)

### Slide 9: Dữ liệu thực nghiệm

**"Hệ thống được kiểm tra trên 4 bộ dữ liệu chuẩn mực:"**

* **MedQA:** Đề thi cấp phép hành nghề y của Mỹ (USMLE) - rất khó.
* **PubMedQA:** Câu hỏi nghiên cứu (Yes/No/Maybe).
* **Vietnamese QA:** Dữ liệu thực tế tại bệnh viện Việt Nam.
* **VQA-RAD:** Hỏi đáp trên hình ảnh X-quang/CT.

### Slide 10: Kết quả Baseline

**"Kết quả ban đầu (chưa có RAG):"**

* Chạy trên Edge (M4 chip) đạt **57.5%** độ chính xác trên MedQA.
* Tốc độ ~2 giây/câu trả lời.
**=> Trả lời RQ1: Hoàn toàn khả thi để chạy trên thiết bị cá nhân.**

### Slide 11: So sánh hiệu quả RAG

**"Khi áp dụng RAG, điều bất ngờ đã xảy ra:"**

* **Q&A Format (Màu đỏ):** Độ chính xác **GIẢM 5%**.
* **Pure Knowledge (Màu xanh):** Độ chính xác **TĂNG 4%** (lên 59%).

---

## 🔑 Phân tích chuyên sâu (Slide 12-14)

### Slide 12: KEY FINDING - "Format > Size"

**"Đây là phát hiện quan trọng nhất của đề tài: ĐỊNH DẠNG dữ liệu quan trọng hơn KÍCH THƯỚC dữ liệu."**

### Slide 13: Tại sao Q&A thất bại?

**"Tôi xin giải thích vì sao đưa thêm dữ liệu Q&A lại làm máy kém đi:"**

* Nó làm máy bị nhiễu: Máy tìm thấy các câu hỏi *tương tự* nhưng *không phải* là câu hỏi đang hỏi.
* Các đáp án sai trong dữ liệu Q&A làm máy bối rối (conflicting signals).

### Slide 14: Tại sao Pure Facts thành công?

**"Ngược lại, Pure Facts thành công vì:"**

* Nó cung cấp **nguyên liệu sạch** (kiến thức chuẩn) để máy tự suy luận.
* Không chứa các yếu tố gây nhiễu.

---

## ⚙️ Mở rộng & Kết luận (Slide 15-20)

### Slide 15: VQA-RAD (Vision)

**"Ngoài văn bản, hệ thống còn đọc được ảnh y tế. Thử nghiệm trên X-quang và CT não cho độ chính xác 100% trên tập mẫu nhỏ."**

### Slide 16: So sánh Edge vs Cloud

**"Tổng kết lại cuộc chiến Edge và Cloud:"**

* **Edge (Chúng tôi):** 59% chính xác. Thua Cloud (85%).
* **NHƯNG đổi lại:** Bảo mật tuyệt đối, không tốn tiền, chạy khi mất mạng.
* **Ứng dụng:** Phù hợp làm trợ lý tra cứu nhanh, sàng lọc ban đầu tại tuyến cơ sở.

### Slide 17: Trả lời câu hỏi nghiên cứu

**"Quay lại 3 câu hỏi ban đầu, chúng tôi đã có câu trả lời khẳng định cho cả 3."** (Chỉ vào slide).

### Slide 18: Cấu hình khuyến nghị

**"Để triển khai, các bệnh viện chỉ cần trang bị máy tính có RAM 16GB, cài đặt theo cấu hình này là có thể sử dụng ngay."**

### Slide 19: Hạn chế

**"Tất nhiên, vẫn còn khoảng cách về độ thông minh so với GPT-4. Hướng tới chúng tôi sẽ Fine-tune (huấn luyện lại) mô hình trên dữ liệu tiếng Việt để cải thiện điều này."**

### Slide 20: Kết luận

**"Tóm lại, RAG-MedGemma trên Edge là một giải pháp KHẢ THI, TIẾT KIỆM và AN TOÀN cho y tế Việt Nam. Đặc biệt là phát hiện về 'Pure Knowledge RAG' sẽ là hướng đi đúng đắn cho các nghiên cứu sau này."**

**"Xin cảm ơn thầy cô và các bạn đã lắng nghe ạ!"**
