# Dữ liệu OCR

- `raw/`: dữ liệu gốc, giữ nguyên để truy xuất nguồn.
- `processed/`: ảnh và nhãn đã làm sạch, chuẩn hóa.
- `splits/`: danh sách train / validation / test dùng chung giữa các model.

Định dạng nhãn đề xuất: JSONL, mỗi dòng gồm `id`, `image`, `text`, `split`, `language`, `source`. Đường dẫn `image` tính từ thư mục `data/`.

Ví dụ (chỉ minh họa định dạng):

```json
{"id":"sample_001","image":"processed/sample_001.png","text":"Hóa đơn bán hàng","split":"test","language":"vi","source":"internal"}
```

Chia theo tài liệu hoặc nguồn trước khi cắt ảnh để tránh rò rỉ dữ liệu. Không dùng tập test để chọn tham số hay fine-tuning. Dữ liệu thực tế không được đưa vào Git theo cấu hình mặc định.
