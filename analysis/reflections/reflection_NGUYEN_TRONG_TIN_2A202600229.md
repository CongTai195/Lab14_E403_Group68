# Reflection Report - Nguyễn Trọng Tín - 2A202600229

## 1. Vai trò và phạm vi công việc
Trong bài lab này, em phụ trách chính phần **Agent engineering** và tham gia trực tiếp vào các module liên quan đến:
- pipeline trả lời của agent trong [agent/main_agent.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/agent/main_agent.py:1)
- retrieval và reranking trong [agent/retriever.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/agent/retriever.py:1)
- benchmark runner bất đồng bộ trong [engine/runner.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/engine/runner.py:1)
- retrieval metrics trong [engine/retrieval_eval.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/engine/retrieval_eval.py:1)
- multi-judge và performance reporting trong [engine/llm_judge.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/engine/llm_judge.py:1) và [main.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/main.py:1)

Mục tiêu công việc của em là biến hệ thống từ một bản benchmark mang tính placeholder thành một pipeline có thể:
- so sánh được `V1` và `V2`
- tính được retrieval metrics thật
- có multi-judge consensus
- sinh report đúng format yêu cầu của giảng viên
- chạy ổn định trên tập benchmark nhiều test case

## 2. Engineering Contribution

### 2.1. Tách hai phiên bản agent đúng tinh thần đề bài
Một đóng góp quan trọng của em là chuyển hệ thống từ kiểu “chỉ đổi tên V1/V2” sang **hai agent thật**:
- `AgentV1Base`: retrieval yếu hơn, logic cũ hơn, prompt đơn giản hơn
- `AgentV2Optimized`: hybrid retrieval tốt hơn, có reranking heuristic, prompt tốt hơn và final answer có cấu trúc tốt hơn

Việc này được thực hiện ở [agent/main_agent.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/agent/main_agent.py:121) và [agent/main_agent.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/agent/main_agent.py:145), kết hợp với benchmark orchestrator trong [main.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/main.py:11).

Ý nghĩa kỹ thuật của phần này là hệ thống regression không còn so sánh hai “label” giả lập nữa, mà so sánh hai pipeline có khác biệt thực sự về retrieval, prompt và answer strategy.

### 2.2. Nâng cấp retrieval và đánh giá retrieval
Em tham gia trực tiếp vào việc tổ chức lại retrieval theo hai mức:
- V1 dùng retrieval `sparse`
- V2 dùng retrieval `hybrid` kết hợp dense + BM25 + rerank

Ở [agent/retriever.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/agent/retriever.py:228), em bổ sung logic hỗ trợ nhiều chế độ retrieval và thêm bước rerank để cải thiện thứ tự candidate cho V2.

Ngoài ra, em nối benchmark với [engine/retrieval_eval.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/engine/retrieval_eval.py:6) để hệ thống không còn dùng số `hit_rate` và `mrr` giả lập mà tính thật từ:
- `expected_retrieval_ids` trong `golden_set.jsonl`
- `retrieved_ids` do agent trả về

Đây là phần đóng góp quan trọng vì nó giúp benchmark đo được mối liên hệ giữa retrieval quality và answer quality thay vì chỉ nhìn mỗi final answer.

### 2.3. Xây dựng Multi-Judge consensus
Em tham gia nâng cấp phần judge từ mock sang **2 OpenAI judge models**:
- `gpt-4o-mini`
- `gpt-4.1-nano`

Ở [engine/llm_judge.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/engine/llm_judge.py:15), em xây dựng cơ chế:
- chấm độc lập bởi 2 judge
- lưu `individual_results`
- tính `agreement_rate`
- gắn `status = consensus/conflict`
- tính `final_score` từ hai judge

Đây là phần engineering có độ phức tạp cao vì vừa liên quan tới async, API integration, parser cho structured output, vừa liên quan tới reliability của hệ thống đánh giá.

### 2.4. Chuẩn hóa report và release gate
Em cũng tham gia chỉnh lại `summary.json` và `benchmark_results.json` ở [main.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/main.py:220) để:
- đúng format giảng viên yêu cầu
- có phần `regression`
- có thêm số liệu `performance`, `token usage`, `estimated cost`

Phần này giúp kết quả benchmark không chỉ “chạy được” mà còn đủ để báo cáo, nộp bài và dùng làm căn cứ giải thích kỹ thuật.

## 3. Technical Depth

### 3.1. MRR là gì và em đã áp dụng như thế nào
`MRR` (Mean Reciprocal Rank) là chỉ số đo xem **kết quả liên quan đầu tiên** xuất hiện ở vị trí nào trong danh sách truy xuất.  
Nếu tài liệu đúng nằm ở:
- vị trí 1 thì MRR = 1.0
- vị trí 2 thì MRR = 0.5
- vị trí 3 thì MRR = 0.333...

Trong project này, em dùng MRR để đánh giá retrieval trong [engine/retrieval_eval.py](/abs/c:/Users/TRONG_TIN/Desktop/AI20K/LAB/Lab14_E403_Group68/engine/retrieval_eval.py:24). Ý nghĩa thực tế là:
- Hit Rate cho biết có tìm thấy tài liệu đúng hay không
- MRR cho biết tài liệu đúng có đứng đủ cao để model dễ dùng hay không

Theo em, MRR đặc biệt quan trọng trong RAG vì nếu chunk đúng đứng quá thấp thì dù retrieval technically “trúng”, answer quality vẫn có thể giảm do context window ưu tiên các chunk đứng trước.

### 3.2. Cohen’s Kappa là gì
`Cohen’s Kappa` là thước đo mức độ đồng thuận giữa hai người chấm hoặc hai judge **sau khi đã loại bớt phần trùng khớp ngẫu nhiên**.  
Khác với agreement rate thông thường, Kappa phản ánh đúng hơn việc hai judge có thực sự đồng thuận hay không.

Trong project hiện tại em mới dừng ở `agreement_rate`, chưa triển khai `Cohen’s Kappa` thật. Tuy nhiên em hiểu vai trò của nó:
- nếu chỉ nhìn phần trăm agreement đơn giản thì có thể đánh giá quá lạc quan
- Kappa phù hợp hơn khi muốn đo độ tin cậy của multi-judge ở mức nghiêm túc hơn

Nếu có thêm thời gian, em sẽ dùng Kappa để calibrate hai judge thay vì chỉ dùng công thức agreement đơn giản như hiện tại.

### 3.3. Position Bias là gì
`Position Bias` là hiện tượng judge có thể thiên vị response đứng trước hoặc đứng sau khi so sánh nhiều câu trả lời.  
Ví dụ, cùng một nội dung nhưng nếu đặt ở vị trí A thì được chấm cao hơn vị trí B, điều này làm kết quả đánh giá không còn khách quan.

Trong project, `position bias` chưa được implement đầy đủ, nhưng em hiểu rằng nếu sau này hệ thống cần so sánh nhiều candidate answer hoặc nhiều version answer cùng lúc thì phải:
- đảo vị trí response
- chấm lại
- kiểm tra xem điểm có thay đổi đáng kể hay không

### 3.4. Trade-off giữa chi phí và chất lượng
Trong quá trình làm project, em nhận thấy rất rõ trade-off giữa chi phí và chất lượng:
- model mạnh hơn cho answer tốt hơn nhưng chậm hơn và tốn token hơn
- concurrency cao giúp benchmark nhanh hơn nhưng dễ dính rate limit
- multi-judge tăng độ tin cậy nhưng làm cost tăng đáng kể
- retrieval mạnh hơn như hybrid + rerank giúp tăng quality nhưng làm latency tăng

Điều này thể hiện rõ trong `summary.json` khi:
- V2 có score tốt hơn V1
- nhưng runtime, token usage và estimated cost của V2 cũng cao hơn

Theo em, đây là trade-off rất thực tế trong AI Engineering: hệ thống tốt không chỉ cần đúng, mà còn phải có chi phí chấp nhận được và chạy ổn định.

## 4. Problem Solving

Trong quá trình làm bài, em gặp nhiều vấn đề kỹ thuật phát sinh và đã trực tiếp xử lý:

### 4.1. Lỗi `chromadb` và phụ thuộc môi trường
Hệ thống ban đầu bị lỗi import liên quan đến `chromadb` và `opentelemetry`, khiến retriever crash ngay từ lúc khởi tạo.  
Em đã thêm hướng fallback để nếu dense store không khởi tạo được thì hệ thống vẫn có thể chạy với BM25-only, tránh việc benchmark dừng hoàn toàn.

### 4.2. Lỗi không tương thích model khi đổi model generate
Khi đổi sang model nhẹ hơn để benchmark nhanh hơn, tham số `reasoning.effort` không còn được hỗ trợ.  
Em đã chỉnh logic để chỉ truyền `reasoning` khi model thuộc nhóm GPT-5, còn các model nhẹ thì dùng request đơn giản hơn.

### 4.3. Rate limit và quota issues
Khi tăng concurrency để benchmark nhanh hơn, hệ thống gặp:
- `429 rate limit`
- quota issues với Gemini

Em đã giải quyết bằng cách:
- giảm batch size
- thêm retry/backoff
- thay judge phụ bằng một OpenAI model khác nhẹ hơn để ổn định hơn về quota

### 4.4. Lỗi proxy và kết nối API
Trong quá trình test thực tế, benchmark bị lỗi `Connection refused` dù máy vẫn có mạng. Em kiểm tra và phát hiện biến môi trường proxy đang trỏ tới `127.0.0.1:9`, làm toàn bộ OpenAI request đi sai đường.  
Việc tự kiểm tra `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` và xác minh kết nối thật tới `api.openai.com` giúp em xác định được lỗi nằm ở môi trường chứ không phải logic code.

### 4.5. Lỗi console encoding trên Windows
Khi chạy `python main.py`, chương trình có thể dừng ngay vì Windows console không in được emoji do `cp1252`. Em nhận ra đây không phải lỗi benchmark mà là lỗi encoding đầu ra. Đây là một ví dụ cho thấy debug hệ thống AI không chỉ là sửa model hay prompt, mà còn phải xử lý cả vấn đề runtime environment.

## 5. Tự đánh giá
Em tự đánh giá phần đóng góp của mình phù hợp mạnh nhất với ba tiêu chí trong rubric cá nhân:

- **Engineering Contribution:** em đã tham gia trực tiếp vào các module phức tạp như agent pipeline, async runner, retrieval metrics, multi-judge và reporting.
- **Technical Depth:** em hiểu và có thể giải thích các khái niệm MRR, agreement, Cohen’s Kappa, Position Bias, cũng như trade-off giữa chất lượng, tốc độ và chi phí.
- **Problem Solving:** em đã xử lý nhiều vấn đề thực tế phát sinh trong quá trình tích hợp hệ thống, từ lỗi dependency, API, quota, rate limit đến môi trường chạy trên Windows.

Nếu có thêm thời gian, em muốn cải thiện tiếp:
- dùng semantic reranker thật thay cho heuristic rerank
- bổ sung calibration sâu hơn cho judge bằng Cohen’s Kappa
- thêm kiểm tra position bias
- làm out-of-scope guardrail riêng trước bước generation

## 6. Kết luận
Qua bài lab này, em học được rằng xây một hệ thống benchmark AI không chỉ là “gọi model và lấy điểm”, mà là sự kết hợp giữa:
- thiết kế agent
- retrieval evaluation
- multi-judge reliability
- performance/cost tracking
- và xử lý các lỗi hệ thống ngoài dự kiến

Phần em phụ trách giúp em hiểu rõ hơn cách một hệ thống RAG hoạt động trong bối cảnh kỹ thuật thực tế, nơi độ ổn định, khả năng đo lường và khả năng debug quan trọng không kém chất lượng câu trả lời.
