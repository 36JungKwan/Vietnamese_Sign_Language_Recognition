import os
import json
import asyncio
import collections
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import mediapipe as mp
import sys
import traceback
import yaml
from dotenv import load_dotenv

# Load các biến môi trường từ file .env
load_dotenv()

# Lấy giá trị của biến
api_key = os.getenv("API_KEY")

os.environ['GLOG_minloglevel'] = '2'       # Cấm MediaPipe in log rác C++
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Cấm các log cảnh báo phần cứng

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- THAY ĐỔI: Nhúng Model Factory thay vì VSLModel cứng ---
from model import create_model 
from decoder import TemporalDecoder
from llm_agent import LLMAgent

# --- GLOBAL CONFIG & QUEUES ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

frame_queue = asyncio.Queue(maxsize=30) 
llm_queue = asyncio.Queue()
tts_queue = asyncio.Queue()
active_connections = []

# ==========================================
# CÁC HÀM XỬ LÝ DỮ LIỆU ĐỘNG TÁC (VISION PIPELINE)
# ==========================================
def load_config(config_path="configs/config.yaml"): # Nên dùng đường dẫn tương đối
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
cfg = load_config()

def load_label_map(path="label_map_472.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
        
    # Trường hợp 1: Nếu file JSON lưu dưới dạng List ["An ủi", "Bố", ...]
    if isinstance(raw_map, list):
        idx_to_class = {idx: name for idx, name in enumerate(raw_map)}
        
    # Trường hợp 2: Nếu file JSON lưu dưới dạng Dict {"An ủi": 0, "Bố": 1}
    elif isinstance(raw_map, dict):
        class_to_idx = raw_map.get("root", raw_map)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        
    else:
        raise ValueError("Định dạng file label_map.json không được hỗ trợ!")
        
    return idx_to_class, len(idx_to_class)

def extract_landmarks(results):
    """Trích xuất chuẩn 76 điểm (x, y, z) từ MediaPipe (Phase 2.0)"""
    landmarks = np.zeros((cfg['model']['num_nodes'], 3), dtype=np.float32)
    
    # 1. Pose (Lấy trọn 33 điểm)
    if results.pose_landmarks:
        for i in range(33):
            lm = results.pose_landmarks.landmark[i]
            landmarks[i] = [lm.x, lm.y, lm.z]
            
        # Tính điểm Cổ (Neck) - Điểm thứ 76 (Index 75)
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        landmarks[75] = [
            (l_sh.x + r_sh.x) / 2.0,
            (l_sh.y + r_sh.y) / 2.0,
            (l_sh.z + r_sh.z) / 2.0
        ]
            
    # 2. Left Hand (21 điểm) - Bắt đầu từ index 33
    if results.left_hand_landmarks:
        for i in range(21):
            lm = results.left_hand_landmarks.landmark[i]
            landmarks[33 + i] = [lm.x, lm.y, lm.z]
            
    # 3. Right Hand (21 điểm) - Bắt đầu từ index 54
    if results.right_hand_landmarks:
        for i in range(21):
            lm = results.right_hand_landmarks.landmark[i]
            landmarks[54 + i] = [lm.x, lm.y, lm.z]
            
    return landmarks

def build_tensor(frame_buffer):
    """Chuyển đổi buffer 48 frames thành Tensor 9 channels (x,y,z, vx,vy,vz, ax,ay,az)"""
    data = np.stack(frame_buffer, axis=0) # (48, 76, 3)

    # Lấy tọa độ của Node 75 (Cổ) ở toàn bộ 48 frames. Shape sẽ là (48, 1, 3)
    neck_coords = data[:, 75:76, :] 
    # Trừ tất cả các điểm cho tọa độ cổ để ép Cổ về gốc tọa độ (0, 0, 0)
    data = data - neck_coords
    
    velocity = np.zeros_like(data)
    velocity[1:] = data[1:] - data[:-1]
    velocity[0] = velocity[1] 
    
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]
    
    combined = np.concatenate([data, velocity, acceleration], axis=-1) # (48, 76, 9)
    # (48, 76, 9) -> (9, 48, 76) -> (1, 9, 48, 76)
    tensor = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) 
    return tensor.to(DEVICE)

# ==========================================
# WORKERS BẤT ĐỒNG BỘ
# ==========================================
async def vision_worker():
    print("[Worker] Vision Worker Started. Đang kiểm tra file...", flush=True)
    
    label_map_path = "label_map_472.json"
    
    # ⚠️ LƯU Ý QUAN TRỌNG: Bạn nhớ cập nhật đường dẫn checkpoint_path này 
    # trỏ vào đúng thư mục động mà file train.py vừa sinh ra nhé!
    # Ví dụ: checkpoints/stgcn_bilstm/run_baseline_1_20260915_1430/best_vsl_model.pth
    checkpoint_path = r"checkpoints/best_vsl_model.pth" 
    
    try:
        idx_to_class, num_classes = load_label_map(label_map_path)
        
        # --- THAY ĐỔI: Khởi tạo tự động qua Model Factory ---
        model = create_model(cfg, num_classes).to(DEVICE)
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Tự động in ra tên model đang chạy để bạn dễ kiểm tra
        model_name = cfg['experiment']['model_name'].upper()
        print(f"✅ Đã nạp xong trọng số {model_name}! (Độ chính xác Val: {checkpoint.get('val_acc', 0):.4f})", flush=True)
        
        decoder = TemporalDecoder(
            idx_to_class, 
            alpha=0.5, 
            conf_thresh=0.5,      
            motion_thresh=0.005,
            end_frames_thresh=15  # Đã tăng độ trễ ngắt câu lên ~0.5 giây
        )
        frame_buffer = collections.deque(maxlen=cfg['data']['sequence_length'])
        
        print("⏳ Đang khởi tạo MediaPipe (Mất khoảng 2-3 giây)...", flush=True)
        
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            print("✅ MediaPipe đã sẵn sàng! Đang chờ Camera...", flush=True)
            
            while True:
                jpeg_bytes = await frame_queue.get()
                
                try:
                    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None: 
                        frame_queue.task_done()
                        continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    
                    hand_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                    landmarks = extract_landmarks(results)
                    frame_buffer.append(landmarks)
                    
                    if len(frame_buffer) == cfg['data']['sequence_length']:
                        input_tensor = build_tensor(list(frame_buffer))
                        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                            logits = model(input_tensor)
                            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                        
                        motion_energy = np.mean(np.abs(landmarks[33:75] - frame_buffer[-2][33:75]))
                        
                        # --- RADAR LOG (Mở ra khi test để căn góc/tốc độ múa) ---
                        top_word = idx_to_class[np.argmax(probs)]
                        # print(f"👀 [Radar] Motion: {motion_energy:.4f} | Hand: {hand_detected} | Đoán: {top_word} ({np.max(probs)*100:.1f}%)", flush=True)
                        # --------------------------------------------------------

                        new_word, sentence = decoder.process_window(probs, motion_energy, hand_detected)
                        
                        if new_word:
                            print(f"[AI] ⚡ Đoán được từ: {new_word}", flush=True)
                        if sentence:
                            print(f"[AI] 🟢 Ngắt câu! Gửi lên LLM: {sentence}", flush=True)
                            await llm_queue.put(sentence)
                            
                except Exception as e:
                    print(f"❌ [LỖI FRAME]: {e}", flush=True)
                
                frame_queue.task_done()
                await asyncio.sleep(0.001) 

    except Exception as e:
        print(f"\n❌❌❌ [FATAL ERROR]: {e}", flush=True)
        traceback.print_exc()

async def llm_worker():
    print("[Worker] LLM Worker Started.")
    # (Khuyên dùng: Nên đưa API key vào biến môi trường os.environ trong tương lai)
    agent = LLMAgent(api_key=api_key) 
    
    while True:
        sentence_words = await llm_queue.get()
        print(f"[LLM] Đang xử lý: {sentence_words}...")
        
        final_sentence = await asyncio.to_thread(agent.process_sentence, sentence_words)
        print(f"[LLM] ✅ Dịch hoàn chỉnh: {final_sentence}")
        
        await tts_queue.put(final_sentence)
        llm_queue.task_done()

async def tts_worker():
    print("[Worker] TTS Worker Started.")
    while True:
        final_sentence = await tts_queue.get()
        
        for connection in active_connections:
            try:
                await connection.send_json({
                    "type": "final_sentence",
                    "text": final_sentence
                })
            except Exception as e:
                print(f"[WebSocket] Lỗi gửi dữ liệu: {e}")
                
        tts_queue.task_done()

# ==========================================
# ENDPOINT & LIFESPAN (Chuẩn mới FastAPI)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động các worker khi Server bật
    tasks = [
        asyncio.create_task(vision_worker()),
        asyncio.create_task(llm_worker()),
        asyncio.create_task(tts_worker())
    ]
    yield
    # Dọn dẹp khi Server tắt (nếu cần)
    for task in tasks:
        task.cancel()

app = FastAPI(title="VSL Realtime Pipeline", lifespan=lifespan)

@app.websocket("/ws/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"🔌 [WebSocket] Client đã kết nối!")
    try:
        while True:
            data = await websocket.receive_bytes()
            if frame_queue.full():
                try:
                    _ = frame_queue.get_nowait() 
                except asyncio.QueueEmpty:
                    pass
            await frame_queue.put(data)
            
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"🔌 [WebSocket] Client ngắt kết nối.")
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"❌ [WebSocket LỖI SERVER]: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)