# VSL Pipeline

Pipeline thử nghiệm nhận diện ngôn ngữ ký hiệu Việt Nam (VSL) theo thời gian thực. Hệ thống nhận video từ webcam, trích xuất keypoint bằng MediaPipe Holistic, phân loại chuỗi động tác bằng mô hình PyTorch và gửi câu nhận diện được tới LLM để hậu xử lý.

## Luồng xử lý

```text
Webcam client
    -> WebSocket /ws/stream
    -> MediaPipe Holistic
    -> 76 keypoint x 9 kênh (tọa độ, vận tốc, gia tốc)
    -> ST-GCN / Transformer / BiLSTM / ST-TR
    -> TemporalDecoder
    -> Gemini LLM
    -> câu tiếng Việt hoàn chỉnh
```

## Cấu trúc chính

- `modules/app.py`: FastAPI server, xử lý video và suy luận thời gian thực.
- `modules/clients.py`: client webcam gửi frame qua WebSocket.
- `modules/train.py`: huấn luyện, validation và benchmark mô hình.
- `modules/dataset.py`: đọc metadata và các file keypoint `.npy`.
- `modules/model.py`: các kiến trúc và model factory.
- `modules/decoder.py`: gom các dự đoán theo thời gian thành câu.
- `modules/llm_agent.py`: tích hợp Gemini để hoàn thiện câu.
- `configs/config.yaml`: cấu hình thí nghiệm, dữ liệu và huấn luyện.
- `modules/label_map_472.json`: ánh xạ nhãn lớp.

## Yêu cầu

- Python 3.10+
- Webcam
- GPU CUDA được khuyến nghị khi huấn luyện hoặc chạy realtime
- Bộ dữ liệu keypoint và file metadata tương ứng
- Gemini API key nếu dùng bước hậu xử lý LLM

Các thư viện chính được sử dụng: PyTorch, NumPy, OpenCV, MediaPipe, FastAPI, Uvicorn, WebSockets, PyYAML, TensorBoard, `thop`, `tqdm` và Google GenAI SDK.

Có thể cài nhanh bằng lệnh sau:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio numpy opencv-python mediapipe fastapi uvicorn websockets pyyaml tensorboard thop tqdm google-genai
```

## Chuẩn bị dữ liệu

Cập nhật `configs/config.yaml` để trỏ tới:

- `data.base_dir`: thư mục `keypoints_splited`, chứa các thư mục `train/` và `test/`.
- `data.json_meta`: file metadata có các trường `split`, `gloss` và `videoid`.

Mỗi file keypoint được kỳ vọng ở dạng:

```text
<base_dir>/<split>/<gloss>/<videoid>.npy
```

Mỗi mẫu có dạng `(T, 76, 3)` và sẽ được nội suy về 48 frame. Hệ thống chuẩn hóa theo keypoint cổ, sau đó tạo 9 kênh gồm tọa độ, vận tốc và gia tốc.

## Huấn luyện

Trong `configs/config.yaml`, chọn `experiment.model_name` và cập nhật các đường dẫn dữ liệu trước khi chạy. Các model được hỗ trợ được ghi ngay trong file cấu hình.

```powershell
python modules/train.py
```

Kết quả huấn luyện, benchmark và TensorBoard được lưu dưới thư mục checkpoint theo model và run. File `label_map_472.json` được tạo lại từ metadata khi bắt đầu huấn luyện.

## Chạy realtime

1. Đặt checkpoint suy luận tại `modules/checkpoints/best_vsl_model.pth` hoặc cập nhật `checkpoint_path` trong `modules/app.py`.
2. Cấu hình khóa Gemini trong biến môi trường (sau khi thay phần truyền khóa trực tiếp trong `modules/app.py` bằng `LLMAgent()`):

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

3. Chạy server và client ở hai terminal riêng:

```powershell
python modules/app.py
python modules/clients.py
```

Server mở WebSocket tại `ws://localhost:8000/ws/stream`. Trong cửa sổ webcam, nhấn `q` để thoát.

> Lưu ý: các script hiện tại có một số đường dẫn tương đối và tuyệt đối phụ thuộc vào môi trường Windows. Nếu chạy từ thư mục khác, hãy kiểm tra lại `config_path`, đường dẫn label map và checkpoint trong mã nguồn.

## TensorBoard

```powershell
tensorboard --logdir checkpoints
```

## An toàn khóa API

Không commit khóa API hoặc dữ liệu huấn luyện vào Git. `modules/llm_agent.py` đã hỗ trợ biến môi trường `GEMINI_API_KEY`, nhưng `modules/app.py` hiện vẫn truyền khóa trực tiếp khi khởi tạo agent; cần loại bỏ khóa đó trước khi chia sẻ repo. Nếu một khóa đã từng xuất hiện trong mã nguồn hoặc lịch sử Git, hãy thu hồi và tạo khóa mới.