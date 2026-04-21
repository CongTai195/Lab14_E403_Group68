# Báo cáo Cá nhân - Lab 14

**Họ và tên:** Đào Văn Sơn  
**Vai trò trong nhóm:** Nhóm Data & RAG Engineer (Xây dựng Data Pipeline, Retrieval System)

---

## 1. Engineering Contribution (Đóng góp Kỹ thuật)
Trong Lab 14 này, tôi chịu trách nhiệm chính trong việc xây dựng và chuẩn hóa "hệ tri thức" (Knowledge Base) cho Agent, cụ thể là kho dữ liệu chính sách của Facebook, chi tiết như sau:

*   **Crawl và Thu thập dữ liệu:** Viết script tự động crawl các trang chính sách của Facebook, trích xuất (extract) nội dung văn bản thô (raw text), bỏ qua các thẻ HTML rác, navigation bar và footer không cần thiết. Đầu ra được lưu dưới dạng các file `.txt` phân tách theo chủ đề (ví dụ: `01_terms_privacy.txt`, `02_community_standards.txt`, v.v.).
*   **Tiền xử lý và Chuẩn hóa (Data Preprocessing):** Thực hiện làm sạch văn bản (Text Cleansing), gỡ bỏ các ký tự ẩn (invisible characters), các chuỗi dòng trống liên tiếp bằng Regex để đảm bảo dữ liệu đưa vào Embedding Model là sạch nhất, giúp LLM không bị nhiễu thông tin (noise).
*   **Chunking Strategy (Chiến lược chia nhỏ văn bản):** Áp dụng kỹ thuật `RecursiveCharacterTextSplitter` có tinh chỉnh. Không chỉ chia theo độ dài cố định, tôi còn đảm bảo không cắt ngang giữa các câu (overlap hợp lý) và bảo toàn thẻ Heading (tiêu đề mục) để giữ trọn vẹn ngữ cảnh ngữ nghĩa (semantic context). Gắn đầy đủ metadata (`chunk_id`, `source_file`, `title`) cho từng đoạn.
*   **Embedding & Vector Database:** Sử dụng mô hình `text-embedding-3-large` nhúng dữ liệu đã chuẩn hóa vào cơ sở dữ liệu Vector để phục vụ cho Dense Retrieval.
*   **Xây dựng bộ máy Hybrid Retriever:** Viết mã cho phần tìm kiếm để Agent có thể gọi đa hình thức:
    *   Tích hợp tìm kiếm theo cụm từ (Sparse Search/BM25) và tìm kiếm theo ngữ nghĩa (Dense Search).
    *   Tính toán kết hợp điểm số bằng phương pháp **RRF (Reciprocal Rank Fusion)** để Agent trả về các Chunk xịn nhất. 

*(Ghi chú: Đóng góp của tôi có thể kiểm chứng qua các đoạn code xử lý pipeline và trong kết quả cấu trúc logs `hybrid_bm25_dense_rrf` ở phần benchmark).*

---

## 2. Technical Depth (Chiều sâu Kỹ thuật)
Qua quá trình thực hiện Lab 14, tôi hiểu sâu hơn về kiến trúc hệ thống RAG và các chỉ số đo lường như sau:

*   **Tầm quan trọng của Chunk Size & Overlap:** Ban đầu khi tôi để chunk size quá lớn (ví dụ >1000 tokens), điểm **Hit Rate@3** thấp vì ngữ cảnh bị loãng, Vector sinh ra không tập trung vào keyword của câu hỏi. Khi tinh chỉnh xuống kích thước vừa đủ (tầm 500-600 ký tự) với overlap nhỏ (50 ký tự) kết hợp kỹ thuật chia theo dấu phân cách tự nhiên (`\n\n`), thông tin tìm được chính xác rõ rệt qua hệ thống đo đạc Ragas.
*   **MRR (Mean Reciprocal Rank):** Tôi đã hiểu cách thức hoạt động của MRR để đánh giá bộ Retriever. Việc tài liệu đúng xuất hiện ở vị trí số 1 (Rank 1) mang lại giá trị hoàn toàn khác so với nằm ở vị trí số 3 (Rank 3) khi gọi LLM. Một bộ Retriever tốt phải có MRR tiệm cận 1.0 (Điều này đã được chứng minh với MRR = 1.0 trong bảng log `benchmark_results.json` bài nhóm tôi).
*   **Tại sao phải dùng Hybrid Search (Sparse + Dense):** Mô hình Dense Embedding rất giỏi trong việc hiểu đồng nghĩa (ví dụ: "chính sách" = "quy định"). Nhưng với các từ khóa cực độc, tên riêng hoặc thuật ngữ kỹ thuật, Dense Search thường bị trượt. Tôi cấu hình thêm BM25 (Sparse) với hệ thống Rank Fusion (RRF) để bù trừ lẫn nhau, giúp lấy được Top-K tài liệu chuẩn xác nhất ở mọi hình thức câu hỏi.

---

## 3. Problem Solving (Giải quyết Vấn đề)
Trong quá trình phát triển Agent và chạy Benchmark với số lượng lớn (50 cases), tôi đã gặp và trực tiếp giải quyết các vấn đề gai góc sau:

1. **Vấn đề Rate Limit & API Quota (Lỗi HTTP 429):** 
    *   *Triệu chứng:* Khi chạy đồng thời bằng Async Runner với 50 câu hỏi, hệ thống đâm hàng loạt request lên API của OpenAI/Gemini để thực hiện Embedding và Generation, dẫn tới báo lỗi `429 - RESOURCE_EXHAUSTED` hoặc `insufficient_quota`. Do đó nguyên chuỗi Evaluator & Judge bị crash, điểm Ragas trả về `null`.
    *   *Cách giải quyết:* Tôi đã triển khai logic **Heuristic Fallback** và cơ chế **Exponential Backoff (Thử lại với độ trễ tăng dần)**. Nếu lỗi Quota/Rate Limit xảy ra, Pipeline sẽ không sập mà sẽ trả về một `heuristic_fallback` answer an toàn, đồng thời tự động gọi Sparse Keyword Retriever thay vì ép gọi Dense Embedding (như được thể hiện trong log `retrieval_method: dense_failed_sparse_fallback`).

2. **Vấn đề "Rác" (Noisy) trong Crawler Data:**
    *   *Triệu chứng:* Dữ liệu thô từ Facebook policy chứa nhiều đoạn menu điều hướng lặp lại (vd: "Home", "Privacy Center", "Newsroom") sinh ra các Vector vô nghĩa. Khi Similarity Search, LLM hay nhặt rác này thay vì nội dung chính.
    *   *Cách giải quyết:* Viết một bộ filter (`Regex`, `BeautifulSoup`) chặn cứng các CSS Class của Header và Footer, chỉ extract nội dung trong thẻ `<article>` hay `<div class="policy-content">`. Từ đó Hit Rate cải thiện đáng kể. 

Qua các thách thức trên, tôi nhận ra Retrieval không chỉ gọi API là xong, mà phụ thuộc 80% vào nỗ lực làm sạch dữ liệu ban đầu!
