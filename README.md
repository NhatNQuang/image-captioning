# Ứng dụng Image Captioning Song ngữ (Anh - Việt)

Đây là một ứng dụng Image Captioning đơn giản nhưng hiệu quả, cho phép người dùng tải lên hình ảnh hoặc chọn ảnh mẫu để sinh ra các mô tả bằng tiếng Anh và tiếng Việt. Ứng dụng cũng hỗ trợ đánh giá chất lượng mô tả trên một bộ ảnh thử nghiệm.

## 1. Lý do chọn mô hình

Dự án này sử dụng các mô hình Vision-Language Models (VLMs) tiên tiến như **BLIP-2**, **OFA**, và **GIT** từ Hugging Face.

* **Có checkpoint sẵn trên Hugging Face:** Tất cả các mô hình được chọn đều có các checkpoint đã được huấn luyện sẵn và dễ dàng truy cập thông qua thư viện `transformers` của Hugging Face, giúp tiết kiệm thời gian và tài nguyên huấn luyện.
* **Hiệu suất mạnh mẽ:** Các mô hình này đã được pre-trained trên lượng dữ liệu khổng lồ, cho phép chúng tạo ra các caption chất lượng cao, chi tiết và ngữ pháp tốt.
    * **BLIP-2:** Đặc biệt mạnh mẽ trong việc học các biểu diễn hình ảnh-ngôn ngữ và được đánh giá cao về khả năng sinh caption.
    * **OFA (One-For-All):** Một mô hình đa nhiệm có khả năng thực hiện nhiều tác vụ VLM, bao gồm captioning, với một kiến trúc thống nhất.
    * **GIT (Generative Image-to-text Transformer):** Được thiết kế đặc biệt để tạo văn bản chất lượng cao từ hình ảnh.
* **Hỗ trợ song ngữ (qua dịch thuật):** Mặc dù các mô hình này chủ yếu sinh caption bằng tiếng Anh, chúng có thể dễ dàng kết hợp với các thư viện dịch thuật (như `deep_translator`) để chuyển đổi caption sang tiếng Việt, đáp ứng yêu cầu song ngữ của dự án.

## 2. Cách chạy ứng dụng

Để chạy ứng dụng, bạn cần thực hiện các bước sau:

1.  **Clone repository:**
    Nếu bạn chưa clone, hãy sử dụng lệnh sau:
    ```bash
    git clone git@github.com:NhatNQuang/image-captioning.git
    cd image-captioning
    ```
    Nếu bạn đã clone và đang ở trong thư mục, bạn có thể bỏ qua bước này.

2.  **Chuẩn bị dữ liệu và ảnh:**
    * Đảm bảo thư mục `test_images/` tồn tại trong thư mục gốc của dự án.
    * **Đặt 10 ảnh thử nghiệm của bạn** vào thư mục `test_images/`. Đảm bảo tên file ảnh trong `captions.xlsx` khớp với tên file ảnh trong thư mục này.
    * Đảm bảo file `captions.xlsx` tồn tại trong thư mục gốc của dự án. File này phải chứa các cột `image_name`, `comment_number` và `comment` với các ground truth captions cho 10 ảnh test của bạn. Mỗi ảnh có thể có nhiều comment.

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Chạy ứng dụng Gradio:**
    ```bash
    python app.py
    ```
    Sau khi chạy, ứng dụng sẽ cung cấp một URL cục bộ (thường là `http://127.0.0.1:7860/`). Mở URL này trong trình duyệt của bạn để sử dụng ứng dụng.

## 3. Đề xuất cải thiện nếu triển khai production

Nếu triển khai ứng dụng này vào môi trường sản phẩm thực tế, có một số điểm có thể được tối ưu hóa:

* **Tối ưu hóa tốc độ và RAM:**
    * **Quantization:** Sử dụng các kỹ thuật lượng tử hóa (ví dụ: FP16 hoặc INT8) cho các mô hình để giảm kích thước và tăng tốc độ inference, đặc biệt khi chạy trên GPU. Thư viện `bitsandbytes` có thể hỗ trợ việc này dễ dàng.
    * **Onnx Runtime / TensorRT:** Chuyển đổi mô hình sang các định dạng tối ưu hóa inference như ONNX hoặc TensorRT để đạt được hiệu suất cao nhất.
    * **Model Pruning / Distillation:** Giảm kích thước mô hình bằng cách loại bỏ các phần không cần thiết hoặc đào tạo một mô hình nhỏ hơn để bắt chước hành vi của mô hình lớn.
* **Dịch vụ dịch thuật:**
    * Thay vì `deep_translator` (có thể bị giới hạn về tần suất hoặc độ ổn định), cân nhắc sử dụng **Google Cloud Translation API chính thức** hoặc **DeepL API** để đảm bảo chất lượng dịch cao hơn, độ tin cậy và khả năng mở rộng. Điều này sẽ yêu cầu một khóa API và có thể phát sinh chi phí.
* **Quản lý mô hình:**
    * Sử dụng một hệ thống quản lý mô hình (ví dụ: MLflow, BentoML) để theo dõi các phiên bản mô hình, dễ dàng triển khai và rollback.
    * Thiết lập một API riêng biệt cho inference mô hình để tách biệt logic backend và frontend, cho phép mở rộng dễ dàng hơn.
* **Xử lý lỗi và logging:** Triển khai logging mạnh mẽ hơn để theo dõi hiệu suất, phát hiện lỗi và debug trong môi trường production.
* **Khả năng mở rộng:** Nếu lượng người dùng tăng, cân nhắc triển khai ứng dụng trên các nền tảng đám mây (AWS, GCP, Azure) với các dịch vụ auto-scaling để đảm bảo ứng dụng luôn khả dụng.
* **Fine-tuning (Tinh chỉnh) mô hình:**
    * Việc **fine-tuning** các mô hình đã chọn (ví dụ: BLIP-2) trên một tập dữ liệu ảnh-caption **cụ thể và chuyên biệt** hơn của bạn (nếu có) có thể cải thiện đáng kể độ chính xác và tính phù hợp của caption với miền dữ liệu đó. Các kỹ thuật như **LoRA (Low-Rank Adaptation)** rất hữu ích cho việc này, giúp tinh chỉnh mô hình lớn với ít tài nguyên hơn và giảm rủi ro overfitting.

---

## 3. Quy trình làm việc với Git Chuyên nghiệp

Bây giờ bạn đã có tất cả các file cần thiết. Chúng ta sẽ áp dụng quy trình làm việc theo nhánh Git mà bạn đã đề xuất.

1.  **Chuyển sang nhánh `develop`:**
    Đây là nơi bạn sẽ thực hiện hầu hết các công việc phát triển.

    ```bash
    git checkout -b develop
    ```

2.  **Thêm tất cả các file vào staged area:**

    ```bash
    git add .
    ```

3.  **Tạo commit đầu tiên cho phần code dự án:**

    ```bash
    git commit -m "feat: Initial setup of image captioning application with BLIP-2, OFA, GIT and evaluation"
    ```
    *Lưu ý:* Tôi đã sử dụng tiền tố `feat:` (feature) cho commit message, đây là một convention phổ biến giúp dễ dàng theo dõi loại thay đổi.

4.  **Đẩy nhánh `develop` lên GitHub:**

    ```bash
    git push -u origin develop
    ```

5.  **Tạo nhánh tính năng (Feature Branch) và làm việc:**
    Bây giờ, mỗi khi bạn muốn phát triển một tính năng mới hoặc sửa lỗi, bạn sẽ tạo một nhánh từ `develop`. Ví dụ, nếu bạn muốn thêm tính năng "upload ảnh tùy ý":

    ```bash
    git checkout -b feature/add-custom-upload
    ```
    Sau đó, bạn sẽ làm việc trên nhánh này, thêm code, commit các thay đổi của mình.

6.  **Hoàn thành tính năng và hợp nhất (Merge):**
    Khi tính năng hoàn thành và đã được kiểm thử trên nhánh tính năng:

    * Đảm bảo bạn đã commit tất cả thay đổi trên nhánh tính năng.
    * Chuyển về nhánh `develop`:
        ```bash
        git checkout develop
        ```
    * Hợp nhất (merge) nhánh tính năng vào `develop`:
        ```bash
        git merge feature/add-custom-upload
        ```
    * Xóa nhánh tính năng (sau khi đã merge thành công):
        ```bash
        git branch -d feature/add-custom-upload
        ```
    * Đẩy nhánh `develop` đã cập nhật lên GitHub:
        ```bash
        git push origin develop
        ```

7.  **Đẩy lên `main` (Production):**
    Khi nhánh `develop` đã ổn định, đã có đủ tính năng và sẵn sàng cho môi trường sản phẩm, bạn sẽ hợp nhất `develop` vào `main`.

    ```bash
    git checkout main
    git merge develop
    git push origin main
    ```
    Thông thường, việc hợp nhất vào `main` được thực hiện qua **Pull Request (hoặc Merge Request)** trên GitHub, cho phép review code và chạy CI/CD pipeline trước khi chính thức đưa vào sản phẩm.

---

Bạn đã có một cấu trúc dự án vững chắc và một quy trình làm việc Git chuyên nghiệp. Bây giờ bạn có thể bắt đầu cài đặt thư viện và chạy ứng dụng của mình.