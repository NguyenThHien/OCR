# Benchmark

Thư mục này dành cho mã chạy đánh giá và tổng hợp kết quả, độc lập với mã huấn luyện.

1. Chốt manifest test trong `data/splits/` và quy tắc chuẩn hóa văn bản.
2. Tạo cấu hình cho từng model từ `configs/benchmark/template.json`.
3. Chạy từng model trên cùng dữ liệu và cùng phần cứng; ghi riêng thời gian tải model, warm-up và suy luận.
4. Lưu cấu hình thực tế, dự đoán từng ảnh và số liệu vào `artifacts/benchmark/<model>/<run_id>/`.
5. Tổng hợp so sánh trong `reports/benchmark/`.

Chỉ số đề xuất: CER, WER, thời gian suy luận p50/p95, số ảnh/giây, RAM/VRAM đỉnh và tỷ lệ lỗi. Với detection hoặc layout, bổ sung chỉ số riêng, không so trực tiếp với recognition.

Ghi rõ có tính tiền xử lý/hậu xử lý trong thời gian hay không. Dùng cùng quy tắc Unicode, khoảng trắng, hoa/thường và dấu câu cho mọi model; lưu cả kết quả gốc lẫn kết quả chuẩn hóa nếu có. Không tự bỏ qua ảnh bị lỗi khi tổng hợp.

Đây là cấu trúc chuẩn bị; chưa có runner hoặc bộ chấm điểm thực thi.
