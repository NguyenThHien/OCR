# OCR cho dự án Hina

Workspace benchmark các model OCR và fine-tuning model được chọn.

## Cấu trúc

| Thư mục | Nội dung |
| --- | --- |
| [data/](./data/README.md) | Dữ liệu gốc, dữ liệu xử lý và tập train/validation/test |
| [models/](./models/README.md) | Tích hợp riêng cho từng model; có mẫu `_template/` |
| [benchmark/](./benchmark/README.md) | Quy trình và mã đánh giá model |
| [finetuning/](./finetuning/README.md) | Quy trình và mã tinh chỉnh model |
| [configs/benchmark/](./configs/benchmark/template.json) | Cấu hình benchmark theo model |
| [configs/finetuning/](./configs/finetuning/template.json) | Cấu hình fine-tuning theo model |
| [artifacts/](./artifacts/) | Trọng số, dự đoán, log và đầu ra mỗi lần chạy; không đưa dữ liệu thực tế vào Git |
| [reports/benchmark/](./reports/benchmark/comparison-template.md) | Bảng so sánh model và phân tích lỗi |
| [reports/finetuning/](./reports/finetuning/experiment-template.md) | Báo cáo trước và sau fine-tuning |
| [scripts/](./scripts/README.md) | Tiện ích dữ liệu dùng chung |

## Bắt đầu

1. Đưa dữ liệu gốc vào `data/raw/`, chuẩn hóa vào `data/processed/` và chuẩn bị manifest theo hướng dẫn dữ liệu.
2. Sao chép `models/_template/` cho mỗi model muốn thử và điền thông tin cài đặt.
3. Sao chép cấu hình mẫu, đặt tên theo model, sau đó tích hợp runner để benchmark trên cùng tập test.
4. Chọn model, tích hợp trainer và fine-tuning bằng train/validation.
5. Benchmark lại trên tập test giữ riêng; điền báo cáo so sánh trước và sau.

Hiện tại đây là bộ khung tổ chức và các tệp mẫu. Chưa cài thư viện, tải model hoặc triển khai mã chạy benchmark/huấn luyện. Cấu hình mẫu chỉ là quy ước dự kiến, cần được runner/trainer của từng model hỗ trợ.
