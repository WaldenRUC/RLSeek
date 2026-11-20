# Kiểm thử tái tạo trên Codex Cloud

Đã chạy `setup_cpu.sh` trong một thư mục cài đặt mới. Phiên bản Python và từng phụ thuộc được ghi trong `provenance.json` đều khớp. Cả hai script Bash qua `bash -n` và cả ba script Python qua kiểm tra cú pháp.

Đã chạy `reproduce.sh` với bản sao sạch của checkout RLSeek tại commit `b4981a2a5b3cbff3d3a64f6ed7507616de704442`. Script tải lại JSONL từ revision đã ghim, xác minh checksum nguồn và tokenizer, rồi tạo liên kết `data/ragtruth` trong checkout kiểm thử. Cả hai tệp Parquet khớp SHA-256 và giống từng byte với lần tạo trước.

| Split | Đầu vào | Giữ lại | Lỗi tokenizer | Vượt 16.384 token | Prompt dài nhất |
|---|---:|---:|---:|---:|---:|
| Train | 15.090 | 15.090 | 0 | 0 | 2.837 |
| Test | 2.700 | 2.700 | 0 | 0 | 2.473 |

`verification.json` ghi kết quả chạy lại với tokenizer thử nghiệm từ gói npm. Loader xử lý và ghép batch toàn bộ 17.790 hàng, so khớp mọi nhãn với dữ liệu gốc và gọi hàm reward cho từng hàng bằng dự đoán rỗng. Số liệu trùng với lần xác minh trước. Tập kiểm tra nhỏ được tạo lại riêng, có 24 mẫu mỗi split và chạy thành công qua `verify_sample.py`. Không chạy inference hay training.
