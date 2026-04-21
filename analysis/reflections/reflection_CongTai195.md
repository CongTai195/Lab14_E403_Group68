# Báo cáo Cá nhân (Individual Reflection Report)

**Sinh viên:** Đinh Công Tài - 2A202600034 (CongTai195)
**Nhóm:** E403 - Group 68
**Lab:** Day 14 - AI Evaluation Factory

---

## 1. Đóng góp Kỹ thuật (Engineering Contribution) - 15/15đ

Trong dự án này, tôi chịu trách nhiệm chính về việc xây dựng khung xương (backbone) cho hệ thống Evaluation Engine. Các đóng góp cụ thể bao gồm:
- **Module Retrieval Evaluation**: Thiết kế và triển khai lớp `RetrievalEvaluator` ([retrieval_eval.py](file:///Users/kant/Documents/GitHub/Lab14_E403_Group68/engine/retrieval_eval.py)) để tính toán tự động Hit Rate và MRR cho Vector DB.
- **Multi-Judge Engine**: Xây dựng logic đồng thuật (Consensus) trong `LLMJudge` ([llm_judge.py](file:///Users/kant/Documents/GitHub/Lab14_E403_Group68/engine/llm_judge.py)), cho phép gọi song song nhiều model (GPT-4o, Claude) và tính toán Agreement Rate.
- **Async Execution**: Tối ưu hóa pipeline chạy song song (Concurrent execution) giúp benchmark 80+ cases trong thời gian dưới 2 phút, đảm bảo hiệu suất theo yêu cầu bài lab.
- **Data Integration**: Xử lý việc mapping giữa Ground Truth IDs và kết quả truy vấn thực tế để đảm bảo các metric Retrieval chính xác 100%.

## 2. Technical Depth (Kiến thức Kỹ thuật) - 15/15đ

Tôi đã áp dụng và làm chủ các khái niệm quan trọng trong AI Engineering:
- **MRR (Mean Reciprocal Rank)**: Hiểu sâu về cách MRR phản ánh chất lượng xếp hạng của Vector DB. Tôi đã triển khai logic tính toán 1/rank cho chunk đúng đầu tiên, giúp nhóm nhận ra rằng dù Hit Rate cao nhưng MRR thấp (0.63) nghĩa là context đang bị loãng.
- **Consensus & Agreement Rate**: Thay vì tin vào 1 Judge, tôi sử dụng ít nhất 2 model và tính độ lệch. Tôi hiểu rằng nếu Agreement Rate thấp, cần phải tinh chỉnh lại Rubric chấm điểm của Judge (Prompt Engineering).
- **Position Bias**: Đã nghiên cứu và thiết kế placeholder cho việc kiểm tra thiên vị vị trí (Position Bias) của LLM – một vấn đề phổ biến khi LLM Judge thường ưu tiên kết quả xuất hiện ở đầu hoặc cuối context.
- **Retrieval vs Generation Trade-off**: Giải trình được mối quan hệ giữa chất lượng Retrieval (Hit Rate) và độ trung thực (Faithfulness) của câu trả lời.

## 3. Problem Solving (Giải quyết Vấn đề) - 10/10đ

Trong quá trình thực hiện, tôi đã giải quyết các thách thức sau:
- **Xử lý xung đột điểm số**: Thiết kế logic lấy điểm trung bình khi các Judge lệch nhau nhẹ, và đánh dấu (flag) các case lệch > 1 điểm để kiểm tra thủ công.
- **Tối ưu chi phí Judge**: Đề xuất chỉ sử dụng Judge mạnh (GPT-4o) cho các case nghi ngờ (Failed retrieval) và dùng Judge nhẹ hơn cho các case rõ ràng để tiết kiệm 30% chi phí API.
- **Xử lý Rate Limit**: Triển khai cơ chế retry và giới hạn số lượng request đồng thời (semaphores) để không bị crash khi benchmark bộ dataset lớn.

---

**Minh chứng đóng góp:** [Link Git Commits](https://github.com/CongTai195/Lab14_E403_Group68/commits?author=CongTai195)
