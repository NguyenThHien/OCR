# OCR benchmark — VinText

## Trạng thái

Đã chạy `PP-OCRv6_small_det` trên 500 ảnh VinText test bằng PaddleOCR. **Chưa chạy model đối chứng `PP-OCRv5_mobile_det`** như prompt đề tài yêu cầu; lệnh bên dưới dùng chung dữ liệu, CPU, batch size và bộ chấm để chạy tiếp model đó. Chưa fine-tune model.

| Chỉ số v6 trên VinText | Kết quả |
| --- | ---: |
| Precision / Recall / H-mean | 27,30% / 7,72% / 12,04% |
| Latency p50 / p95 | 2,05 / 4,29 giây/ảnh |
| Lỗi suy luận | 0 / 500 |

Kết quả đầy đủ, biểu đồ, ảnh lỗi, protocol và phân tích nằm trong [báo cáo benchmark](reports/benchmark/vintext_ppocrv6_small_det.md). Model thường bao một dòng chữ trong khi VinText gán nhãn từng từ; vì vậy H-mean theo IoU từng polygon thấp hơn độ phủ chữ nhìn thấy. Hãy thống nhất đơn vị nhãn trước khi quyết định fine-tune.

## Chạy từng bước

PowerShell tại thư mục gốc repository; môi trường đã kiểm chứng là Python 3.11.9 trên Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt

# Đọc/kiểm tra ảnh và nhãn VinText, tạo manifest cùng audit
.\.venv\Scripts\python.exe -m benchmark.run prepare

# Chạy thử 5 ảnh với v6
.\.venv\Scripts\python.exe -m benchmark.run run --limit 5 --output artifacts/benchmark/PP-OCRv6_small_det/smoke_cpu

# Chạy model đối chứng v5; mặc định cùng cấu hình CPU, manifest, warm-up và metrics
.\.venv\Scripts\python.exe -m benchmark.run run --model PP-OCRv5_mobile_det --output artifacts/benchmark/PP-OCRv5_mobile_det/vintext_cpu

# Model khác: chỉ đổi model name và thư mục đầu ra
.\.venv\Scripts\python.exe -m benchmark.run run --model PP-OCRv6_small_det --output artifacts/benchmark/PP-OCRv6_small_det/vintext_cpu_next
```

Mỗi thư mục đầu ra chỉ chạy một lượt; chương trình không ghi đè run cũ. Khi so tốc độ hai model, chạy ít nhất 3 lượt mỗi model vào thư mục khác nhau rồi đối chiếu median/p95; đừng suy ra độ ổn định thời gian từ lượt baseline đơn.

Baseline v6 đã có trong `artifacts/benchmark/PP-OCRv6_small_det/baseline_cpu_clean/`. Mỗi run cần một thư mục đầu ra mới; chương trình không cho ghi đè run đã claim. Chọn CPU/GPU và model local bằng cách chỉnh `device` và `model_options` trong [cấu hình benchmark](configs/benchmark/ppocrv6_small_det.json). Ví dụ model local: thêm `"model_dir": "C:/duong-dan/model-infer"` vào `model_options`. Trọng số chính thức tự tải lần đầu và cache tại `artifacts/checkpoints/paddlex/`; thiết bị GPU cần cài đúng bản PaddlePaddle GPU tương thích.

## Dữ liệu và cách chấm

VinText test nằm trong `data/raw/datasets/vintext/test/`: ghép `images/im1501.jpg` với `annotations/gt_1501.txt`. Mỗi dòng nhãn gồm 8 tọa độ polygon và transcription; `###` là vùng ignore. Manifest `data/splits/vintext_test.jsonl` được tạo từ ảnh/nhãn gốc, không resize hoặc augmentation ảnh test. Audit lưu số mẫu và SHA256. Hiện chỉ có test, không có train/validation; không dùng test để chọn threshold hoặc checkpoint.

Annotation hiện không có nhãn loại nội dung/điều kiện chụp (mờ, thiếu sáng, nghiêng...). Vì vậy không báo cáo số liệu theo các nhóm đó; chỉ phân tích kích thước polygon và kiểm tra overlay, không tự gán nhãn điều kiện ảnh.

Benchmark ghép polygon một-một ở IoU >= 0,5; tổng hợp micro TP/FP/FN, đồng thời lưu latency p50/p95, throughput, RSS lấy mẫu, lỗi, CSV từng ảnh, ảnh overlay và biểu đồ. Đây là protocol chẩn đoán cục bộ, **không phải evaluator chính thức của VinText**. Model detection không đọc text nên không có CER/WER. Xem rõ phạm vi đo và giới hạn trong báo cáo trước khi so số liệu với nguồn khác.

Code được tách theo trách nhiệm: `benchmark/dataset.py` đọc dữ liệu, `models/detection.py` nạp adapter PaddleOCR theo tên model, `benchmark/metrics.py` chấm polygon, `benchmark/run.py` chạy inference và đo tốc độ, `benchmark/report.py` xuất phân tích/biểu đồ. Muốn thêm model PaddleOCR, truyền model name khác; framework khác chỉ cần adapter có `predict(image)` và `synchronize()`.

Tạo lại report/biểu đồ từ dự đoán đã lưu, không chạy inference lại:

```powershell
.\.venv\Scripts\python.exe -m benchmark.run report --run-dir artifacts/benchmark/PP-OCRv6_small_det/baseline_cpu_clean
```

`artifacts/`, dữ liệu test và `.venv/` được giữ local, không commit. Không cần `.env`; không có secret hoặc biến môi trường bắt buộc trong cấu hình benchmark.
