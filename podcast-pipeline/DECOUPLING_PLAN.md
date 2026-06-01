# Pipeline Decoupling & Notebook Separation Plan

## Mục đích
Hệ thống hiện tại phụ thuộc quá lớn vào file monolith `main_original_ASR_MoE.py`. File này import toàn bộ các thư viện AI nặng (`nemo`, `whisper`, `demucs`, `pyannote`...) ở đầu file. Hậu quả là dù chỉ chạy một module rất nhỏ (như Export MP3), hệ thống cũng bắt buộc phải cài đặt toàn bộ mô hình AI. Điều này gây tốn thời gian cực kỳ lớn khi triển khai trên Kaggle hoặc máy chủ mới.

Mục tiêu của kế hoạch này là:
1. **Refactor giảm Coupling:** Chuyển các thư viện nặng thành "Lazy Import" (chỉ import khi hàm thực sự được gọi).
2. **Tách biệt Pipeline:** Tạo các file Jupyter Notebook độc lập (`nb_stage_01.ipynb` -> `nb_stage_05.ipynb`), mỗi file chỉ cài những thư viện mà stage đó cần. Điều này giúp chạy nhanh, test nhanh, và dễ dàng cải tiến hoặc thay đổi từng stage sau này.

## Giai đoạn 1: Refactor Code (Giảm Coupling)
Thay đổi cơ chế import trong `main_original_ASR_MoE.py` và các module phụ thuộc:
- `nemo`, `pyannote`: Di chuyển vào trong hàm/class liên quan đến Diarization.
- `demucs`, `panns_inference`: Di chuyển vào trong hàm/class xử lý Audio Separation (Stage 2).
- `whisper`, `transformers`, `g2pk`: Di chuyển vào trong hàm/class ASR (Stage 4).
- Mọi module chung (Common utils) phải đảm bảo không import trực tiếp AI Model.

## Giai đoạn 2: Tạo các Jupyter Notebook độc lập
Sau khi `main_original_ASR_MoE.py` đã an toàn để import mà không bị lỗi thiếu thư viện, ta sẽ sinh ra 5 file `.ipynb` tương ứng với 5 công đoạn.

**Cấu trúc một file Notebook chuẩn:**
1. Clone mã nguồn từ Github.
2. Cài đặt môi trường (Minimal Dependencies - chỉ cài những gì thật sự cần).
3. Import và chạy `stage_XX.py`.
4. Nén kết quả (`zip`) và cung cấp link tải về để làm đầu vào cho Notebook của Stage tiếp theo.

## Lợi ích
- **Tiết kiệm thời gian:** Không phải cài lại Whisper nếu chỉ test Diarization.
- **Tiết kiệm tài nguyên:** Kaggle sẽ không bị quá tải VRAM vì không nạp dư thừa thư viện vào RAM.
- **Dễ bảo trì:** Có thể giao việc cải tiến Stage 4 cho một người mà họ không cần bận tâm hệ thống Stage 1 chạy như thế nào.
