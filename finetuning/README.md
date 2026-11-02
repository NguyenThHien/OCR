# Fine-tuning

Mã chuẩn bị dữ liệu huấn luyện và tích hợp trainer đặt ở đây, chia tiếp theo tên model khi đã lựa chọn.

1. Chọn model sau benchmark và xác nhận model cho phép fine-tuning.
2. Chuyển dữ liệu train/validation sang định dạng model yêu cầu, giữ riêng tập test.
3. Tạo cấu hình từ `configs/finetuning/template.json`.
4. Lưu checkpoint, log và cấu hình thực tế vào `artifacts/finetuning/<model>/<run_id>/`.
5. Đánh giá checkpoint tốt nhất bằng quy trình benchmark ban đầu và ghi báo cáo vào `reports/finetuning/`.

Mỗi lần chạy cần ghi model/revision gốc, seed, phiên bản dữ liệu, tham số, phần cứng và tiêu chí chọn checkpoint. Trọng số tải sẵn có thể lưu tại `artifacts/checkpoints/<model>/`.

Đây là cấu trúc chuẩn bị; chưa có trainer thực thi. Các tham số trong cấu hình mẫu cần điều chỉnh theo model và bộ nhớ GPU.
