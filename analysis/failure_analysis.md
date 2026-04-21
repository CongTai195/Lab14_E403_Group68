# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** X/Y
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.XX
    - Relevancy: 0.XX
- **Điểm LLM-Judge trung bình:** X.X / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Hallucination | 5 | Retriever lấy sai context |
| Incomplete | 3 | Prompt quá ngắn, không yêu cầu chi tiết |
| Tone Mismatch | 2 | Agent trả lời quá suồng sã |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: [Mô tả ngắn]
1. **Symptom:** Agent trả lời sai về...
2. **Why 1:** LLM không thấy thông tin trong context.
3. **Why 2:** Vector DB không tìm thấy tài liệu liên quan nhất.
4. **Why 3:** Chunking size quá lớn làm loãng thông tin quan trọng.
5. **Why 4:** ...
6. **Root Cause:** Chiến lược Chunking không phù hợp với dữ liệu bảng biểu.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking.
- [ ] Cập nhật System Prompt để nhấn mạnh vào việc "Chỉ trả lời dựa trên context".
- [ ] Thêm bước Reranking vào Pipeline.
