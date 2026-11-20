# Tái tạo Parquet RAGTruth cho RLSeek

Bản chuyển đổi này đã được kiểm tra với loader và hàm reward của RLSeek. Prompt phát hiện hallucination, cách biểu diễn `reference` và quy ước `binary_label = int(bool(labels))` là các lựa chọn thử nghiệm. Cấu hình preprocessing chính thức của tác giả RLSeek chưa được xác minh.

## Chạy trên Codex Cloud

Từ thư mục gốc RLSeek, dùng máy Linux x86-64 có `uv`, `curl` và `sha256sum`. Lệnh cài đặt chuẩn bị Python 3.10.22, PyTorch CPU và cache tokenizer cần cho hàm reward trong `/workspace/.rlseek`. Đặt `RLSEEK_SETUP_DIR` trước cả hai lệnh nếu cần dùng đường dẫn khác ngoài checkout.

```bash
bash tools/ragtruth/setup_cpu.sh
bash tools/ragtruth/reproduce.sh "$PWD" /workspace/rlseek-ragtruth-rebuild
```

Đối số thứ hai của `reproduce.sh` phải là một thư mục build mới. Script tải `response.jsonl` và `source_info.jsonl` từ commit RAGTruth `c103204b9ce28d6bbad859304bf30de72b8ed8fe`, kiểm tra SHA-256 và ghép theo `source_id`. Nó giữ nguyên thứ tự và split, văn bản phản hồi, ID nguồn và phản hồi, cùng mọi trường trong các nhãn span. Tất cả 17.790 hàng được xuất, kể cả các hàng được gắn cờ chất lượng.

Script xuất `full/{train,test}.parquet` trong thư mục build và kiểm tra SHA-256. Nó tạo liên kết `data/ragtruth` trỏ tới `full/` khi đường dẫn này chưa tồn tại. Nếu đã có dữ liệu, script so sánh hai tệp hiện có với bản vừa tạo từng byte trước khi xác minh bằng loader. Giữ thư mục build mà liên kết đang trỏ tới. Kết quả kiểm tra mới nằm trong `verification.json` và `verification.log` ở thư mục build.

```bash
test -s data/ragtruth/train.parquet && test -s data/ragtruth/test.parquet
cd /workspace/rlseek-ragtruth-rebuild
sha256sum -c /workspace/RLSeek/tools/ragtruth/reports/data.sha256
```

## Phạm vi kiểm tra

`loader-config.yaml` ghi cấu hình dữ liệu thu được khi áp dụng tham số của `run.sh` lên `_generated_ppo_trainer`. Cấu hình mặc định của RLSeek tham chiếu `data/legacy_data.yaml`, tệp đang thiếu trong checkout được kiểm tra. `verify_full.py` sử dụng `RLHFDataset` với giới hạn 16.384 token, đối chiếu từng bản ghi với JSONL và chạy hàm reward. `verify_sample.py` giữ lại quy trình kiểm thử 48 mẫu từ bước khảo sát.

Lần chạy chuẩn dùng các tệp tokenizer JSON đã xác minh checksum từ `@lenml/tokenizer-qwen2_5@3.7.2`. Trình tải của gói này khai báo nguồn Qwen2.5-1.5B-Instruct. Mức tương đương với tokenizer Qwen2.5-7B-Instruct chưa được xác minh. Khi có tokenizer chính thức ở máy cục bộ, có thể chạy lại với `RLSEEK_TOKENIZER_DIR=/path/to/qwen-7b-tokenizer`. Các lệnh kiểm tra chỉ dùng tokenizer và CPU, không tải trọng số hoặc khởi chạy inference hay training.

Báo cáo, commit nguồn, phiên bản thư viện và checksum nằm trong `reports/`. Giữ dữ liệu, tokenizer, cache và môi trường Python ở ngoài checkout. Quy tắc `data/` hiện có trong `.gitignore` bao gồm cả liên kết dữ liệu dùng khi chạy.
