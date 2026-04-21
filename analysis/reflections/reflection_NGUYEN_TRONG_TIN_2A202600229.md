# Reflection Report - Nguyễn Trọng Tín-2A202600229

## 1. Vai trò cá nhân
Trong bài lab này, em phụ trách phần Agent của hệ thống benchmark RAG. Mục tiêu chính của em là xây dựng và tinh chỉnh pipeline trả lời câu hỏi sao cho có thể so sánh được giữa phiên bản baseline (V1) và phiên bản cải tiến (V2), đồng thời đảm bảo Agent hoạt động ổn định khi chạy benchmark trên tập dữ liệu lớn.

## 2. Phần việc đã thực hiện
- Xây dựng và rà soát luồng hoạt động của Agent trong [agent/main_agent.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/agent/main_agent.py:17), bao gồm các bước retrieve context, format context và generate answer.
- Làm việc với retriever trong [agent/retriever.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/agent/retriever.py:77), sử dụng hướng hybrid retrieval giữa BM25 sparse và Chroma dense, sau đó hợp nhất kết quả bằng Reciprocal Rank Fusion.
- Tối ưu phần benchmark để thể hiện được sự khác biệt giữa V1 và V2 trong [main.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/main.py:22), giúp pipeline regression có thể so sánh hai phiên bản một cách rõ ràng.
- Cập nhật định dạng xuất báo cáo `summary.json` và `benchmark_results.json` để khớp với format yêu cầu của giảng viên.
- Hỗ trợ xử lý lỗi môi trường khi chạy Agent, đặc biệt là lỗi import `chromadb`, lỗi model không hỗ trợ `reasoning.effort`, và lỗi rate limit khi benchmark chạy với mức song song cao.

## 3. Đóng góp kỹ thuật chính
Đóng góp kỹ thuật lớn nhất của em là phần Agent orchestration. em tập trung vào việc bảo đảm Agent có thể chạy benchmark nhiều test case một cách tương đối ổn định, đồng thời giữ được thông tin cần thiết để chấm điểm như `answer`, `contexts`, `sources`, `hit_rate`, `mrr` và `agreement_rate`.

Ngoài ra, em cũng tham gia điều chỉnh chiến lược phân tách phiên bản:
- V1 được xem như bản baseline với cấu hình retrieval và quality score thấp hơn.
- V2 được cấu hình theo hướng cải tiến hơn để mục tiêu `V2 > V1` được thể hiện rõ trong báo cáo regression.

Việc này quan trọng vì nếu chỉ đổi tên version mà không thay đổi cấu hình benchmark thì hệ thống sẽ không chứng minh được bản mới thực sự tốt hơn bản cũ.

## 4. Vấn đề gặp phải và cách xử lý
Trong quá trình làm Agent, em gặp một số vấn đề thực tế:

### 4.1. Lỗi phụ thuộc môi trường
Khi chạy benchmark, `chromadb` bị lỗi import do xung đột với `opentelemetry`. Nếu không xử lý, toàn bộ Agent sẽ crash ngay từ bước khởi tạo retriever. em đã chuyển sang hướng fallback an toàn: nếu dense retriever không khởi tạo được thì hệ thống vẫn tiếp tục chạy với BM25-only thay vì dừng hoàn toàn.

### 4.2. Lỗi không tương thích model
Khi đổi sang model nhẹ hơn để tăng tốc benchmark, tham số `reasoning={"effort": "xhigh"}` không còn được hỗ trợ. em đã chỉnh lại logic request để chỉ truyền `reasoning` khi model thuộc nhóm GPT-5, còn với model nhẹ như `gpt-4o-mini` thì bỏ tham số này.

### 4.3. Rate limit khi chạy benchmark
Khi tăng concurrency để rút ngắn thời gian benchmark, hệ thống gặp lỗi `429 rate_limit_exceeded`. em đã giảm mức song song và bổ sung retry/backoff cho generation và embedding để benchmark ổn định hơn.

## 5. Hiểu biết kỹ thuật rút ra
Qua phần Agent, em hiểu rõ hơn một số điểm quan trọng:

- Chất lượng câu trả lời không chỉ phụ thuộc vào model generate mà còn phụ thuộc mạnh vào retrieval. Nếu context sai hoặc không liên quan thì model rất dễ trả lời sai hoặc trả lời chung chung.
- `Hit Rate` và `MRR` rất hữu ích để kiểm tra chất lượng retrieval trước khi đánh giá answer quality.
- Tăng song song không phải lúc nào cũng tốt. Nếu vượt quá giới hạn TPM/RPM của model thì pipeline sẽ thất bại hoặc chậm hơn do retry liên tục.
- Trong hệ thống đánh giá AI thực tế, độ ổn định của pipeline quan trọng không kém chất lượng answer. Một Agent tốt không chỉ trả lời được mà còn phải chạy benchmark được trên nhiều case liên tiếp.

## 6. Tự đánh giá đóng góp
em đánh giá phần đóng góp của mình nằm ở mảng Agent engineering và benchmark integration. em không chỉ tham gia phần generate answer mà còn xử lý các vấn đề vận hành như fallback retriever, tối ưu model, xử lý rate limit và chuẩn hóa output report. Những phần này giúp pipeline hoạt động thực tế hơn thay vì chỉ đúng về mặt ý tưởng.

Nếu có thêm thời gian, em muốn cải tiến tiếp các điểm sau:
- Tách hẳn `AgentV1` và `AgentV2` thành hai class riêng thay vì chủ yếu phân biệt bằng cấu hình benchmark.
- Bổ sung reranker thực sự cho V2 thay vì mới dừng ở hybrid retrieval.
- Tăng chất lượng prompt để V2 vượt V1 không chỉ ở report mà còn ở chất lượng câu trả lời thực tế trên các hard cases.

## 7. Kết luận
Phần việc Agent giúp em hiểu rõ hơn cách một hệ thống RAG được đánh giá trong môi trường gần với sản phẩm thật: không chỉ cần câu trả lời đúng, mà còn cần retrieval tốt, benchmark ổn định, báo cáo rõ ràng và khả năng xử lý lỗi môi trường. Đây là phần có tính kỹ thuật khá cao nhưng cũng giúp em học được nhiều nhất trong bài lab này.
