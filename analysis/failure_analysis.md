# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Phiên bản:** Agent_V2_Optimized
- **Tổng số cases:** 82
- **Tỉ lệ Pass/Fail:** 79/3 (Dựa trên phản hồi "Không đủ căn cứ")
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.93
    - Relevancy: 0.88
- **Điểm LLM-Judge trung bình:** 4.6 / 5.0
- **Chỉ số Retrieval:**
    - Hit Rate: 0.95
    - MRR: 0.63
- **Độ đồng thuận (Agreement Rate):** 0.86

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| No Answer / Information Gap | 3 | Dữ liệu nguồn thiếu hoặc Retriever không lấy được chunk chứa câu trả lời |
| Retrieval Inefficiency | ~4 | MRR thấp (0.63) cho thấy chunk đúng không nằm ở top 1, dẫn đến context bị loãng |
| Judge Disagreement | 11 | Một số câu trả lời mang tính kỹ thuật cao khiến Judge 1 và Judge 2 chấm điểm khác nhau |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Chính sách này không áp dụng cho những ai?
1. **Symptom:** Agent trả lời "Không đủ căn cứ trong kho tài liệu để trả lời chính xác."
2. **Why 1:** LLM không thấy thông tin về các đối tượng bị LOẠI TRỪ trong context.
3. **Why 2:** Vector DB trả về các chunk nói về đối tượng ĐƯỢC CHẤP NHẬN/BẢO VỆ thay vì đối tượng bị loại trừ.
4. **Why 3:** Hệ thống RAG gặp khó khăn với các câu hỏi phủ định (negative constraints).
5. **Why 4:** Embedding model không phân biệt rõ ràng giữa "apply to X" và "does not apply to X" khi tìm kiếm semantics.
6. **Root Cause:** Thiếu bước lọc (filtering) hoặc phân tích thực thể (entity extraction) để xử lý các câu hỏi loại trừ.

### Case #2: Những gì sẽ xảy ra nếu cung cấp ngày sinh không chính xác?
1. **Symptom:** Agent không trả lời được các hình phạt cụ thể.
2. **Why 1:** Chunk retrieved chỉ nói về yêu cầu "tên thật" và "ngày sinh chính xác" mà không nói về hậu quả.
3. **Why 2:** Thông tin về xử phạt có thể nằm ở một document khác (Enforcement) chưa được tối ưu chunking.
4. **Why 3:** Chunking size cố định làm mất mối liên kết giữa "Quy định" và "Hệ quả" nằm ở các đoạn xa nhau.
5. **Why 4:** Agent không có khả năng truy vấn đa bước (multi-step) để tìm thêm thông tin sau khi thất bại lần đầu.
6. **Root Cause:** Chiến lược Chunking chưa tối ưu cho các tài liệu pháp lý có cấu trúc tham chiếu chéo.

### Case #3: Có công ty nào khác sử dụng cookie liên quan đến Sản phẩm của Meta?
1. **Symptom:** Agent bỏ lỡ thông tin về bên thứ ba.
2. **Why 1:** Context chỉ chứa thông tin chung về cookie của Meta.
3. **Why 2:** Retriever bị giới hạn bởi Top-K (K=6) nên các chunk về "Third-party cookies" đứng ở vị trí thấp hơn bị loại bỏ.
4. **Why 3:** MRR thấp dẫn đến việc các thông tin bổ trợ quan trọng không được đưa vào prompt.
5. **Why 4:** Query quá đơn giản làm Vector DB trả về nhiều kết quả trùng lặp nội dung.
6. **Root Cause:** Thiếu cơ chế Reranking để đẩy các thông tin chuyên biệt (bên thứ ba) lên cao hơn các thông tin chung chung.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Nâng cấp Top-K từ 4 lên 6 (Đã thực hiện trong V2).
- [ ] Triển khai **Cohere Reranker** để cải thiện MRR từ 0.63 lên > 0.8.
- [ ] Sử dụng **Semantic Chunking** thay cho Fixed-size để giữ trọn vẹn ngữ cảnh của các điều khoản pháp lý.
- [ ] Bổ sung **Query Expansion** (HyDE hoặc multi-query) để xử lý tốt hơn các câu hỏi về đối tượng bị loại trừ.
- [ ] Tinh chỉnh System Prompt để Agent biết cách tóm tắt thông tin liên quan ngay cả khi không có câu trả lời trực tiếp 100%.
