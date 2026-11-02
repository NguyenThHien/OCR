# Tích hợp model

Mỗi model có một thư mục riêng, chứa hướng dẫn cài đặt và mã tích hợp sau này. Chưa cài framework hoặc tải trọng số trong bước tạo cấu trúc.

Khi thêm model, sao chép `_template/` và đổi tên. Có thể bắt đầu với PaddleOCR, EasyOCR, Tesseract hoặc model OCR đa phương thức; chỉ thêm model thực sự cần đánh giá.

Quy ước đầu ra chung cho mỗi ảnh: `id`, `prediction`, `latency_ms`, `error` (null nếu thành công). Lưu nguyên văn dự đoán để có thể chấm lại bằng cùng một cách chuẩn hóa.

Mỗi model nên có môi trường riêng nếu thư viện hoặc phiên bản CUDA xung đột. Trọng số đặt tại `artifacts/checkpoints/<model>/`, không đặt trong mã nguồn.
