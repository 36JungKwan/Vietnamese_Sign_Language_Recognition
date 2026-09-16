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

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("⚠️ Thư viện 'keyboard' chưa được cài. Chạy 'pip install keyboard' để bắt phím Space trên máy chủ.")

# Load các biến môi trường từ file .env
load_dotenv()

# Lấy giá trị của biến
api_key = os.getenv("API_KEY")

os.environ['GLOG_minloglevel'] = '2'       # Cấm MediaPipe in log rác C++
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Cấm các log cảnh báo phần cứng

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Nhúng Model Factory ---
from model import create_model 
from decoder import TemporalDecoder
from llm_agent import LLMAgent

# --- GLOBAL CONFIG & QUEUES ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_queue = asyncio.Queue(maxsize=30) 
llm_queue = asyncio.Queue()
tts_queue = asyncio.Queue()
active_connections = []

# Trạng thái nhận diện (Được bật/tắt bằng phím Space)
IS_RECORDING = False

def toggle_recording():
    global IS_RECORDING
    IS_RECORDING = not IS_RECORDING
    status_str = "🟢 ĐANG BẬT NHẬN DIỆN (Bắt đầu múa ký hiệu...)" if IS_RECORDING else "🔴 ĐÃ TẮT NHẬN DIỆN (Tạm dừng)"
    print(f"\n[HOTKEY SPACE] {status_str}", flush=True)

# ==========================================
# CÁC HÀM XỬ LÝ DỮ LIỆU ĐỘNG TÁC (VISION PIPELINE)
# ==========================================
def load_config(config_path=r"C:/Users/dotru/STUDIE/Competition/Sang_tao_tre_AI/VSL_pipeline/configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
cfg = load_config()

def load_label_map(path=r"C:\Users\dotru\STUDIE\Competition\Sang_tao_tre_AI\VSL_pipeline\modules\label_map_472.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
        
    if isinstance(raw_map, list):
        idx_to_class = {idx: name for idx, name in enumerate(raw_map)}
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

    neck_coords = data[:, 75:76, :] 
    data = data - neck_coords
    
    velocity = np.zeros_like(data)
    velocity[1:] = data[1:] - data[:-1]
    velocity[0] = velocity[1] 
    
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]
    
    combined = np.concatenate([data, velocity, acceleration], axis=-1) # (48, 76, 9)
    tensor = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) 
    return tensor.to(DEVICE)

# ==========================================
# WORKERS BẤT ĐỒNG BỘ
# ==========================================
async def vision_worker():
    global IS_RECORDING
    print("[Worker] Vision Worker Started. Đang kiểm tra file...", flush=True)
    
    label_map_path = r"C:\Users\dotru\STUDIE\Competition\Sang_tao_tre_AI\VSL_pipeline\modules\label_map_472.json"
    checkpoint_path = r"C:\Users\dotru\STUDIE\Competition\Sang_tao_tre_AI\VSL_pipeline\modules\checkpoints\stgcn_transformer\baseline_20260915_1643\best_vsl_model.pth" 
    
    try:
        idx_to_class, num_classes = load_label_map(label_map_path)
        
        model = create_model(cfg, num_classes).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        model_name = cfg['experiment']['model_name'].upper()
        print(f"✅ Đã nạp xong trọng số {model_name}! (Độ chính xác Val: {checkpoint.get('val_acc', 0):.4f})", flush=True)
        print("💡 [HƯỚNG DẪN]: Nhấn phím SPACE trên bàn phím để BẬT/TẮT nhận diện!", flush=True)
        
        decoder = TemporalDecoder(
            idx_to_class, 
            alpha=0.5, 
            conf_thresh=0.5,      
            motion_thresh=0.005,
            end_frames_thresh=15
        )
        frame_buffer = collections.deque(maxlen=cfg['data']['sequence_length'])
        
        print("⏳ Đang khởi tạo MediaPipe (Mất khoảng 2-3 giây)...", flush=True)
        
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            print("✅ MediaPipe đã sẵn sàng! Đang chờ Camera...", flush=True)
            
            last_recording_state = IS_RECORDING

            while True:
                jpeg_bytes = await frame_queue.get()
                
                try:
                    # Nếu trạng thái chuyển từ bật -> tắt hoặc tắt -> bật, clear buffer để tránh dính frames cũ
                    if last_recording_state != IS_RECORDING:
                        frame_buffer.clear()
                        last_recording_state = IS_RECORDING

                    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None: 
                        frame_queue.task_done()
                        continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    
                    # CHỈ KHI BẬT NHẬN DIỆN (IS_RECORDING == True) THÌ MỚI ĐẨY QUA MODEL
                    if IS_RECORDING:
                        hand_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                        landmarks = extract_landmarks(results)
                        frame_buffer.append(landmarks)
                        
                        if len(frame_buffer) == cfg['data']['sequence_length']:
                            input_tensor = build_tensor(list(frame_buffer))
                            with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                                logits = model(input_tensor)
                                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                            
                            motion_energy = np.mean(np.abs(landmarks[33:75] - frame_buffer[-2][33:75]))
                            new_word, sentence = decoder.process_window(probs, motion_energy, hand_detected)
                            print(f"👀 [Radar] Buffer: {len(frame_buffer)}/48 | Motion: {motion_energy:.4f} | Hand: {hand_detected}", flush=True)
                            
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
# ENDPOINT & LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Đăng ký sự kiện nhấn phím Space toàn hệ thống
    if KEYBOARD_AVAILABLE:
        try:
            keyboard.add_hotkey('space', toggle_recording)
            print("⌨️ [HOTKEY] Đã đăng ký phím 'Space' thành công!")
        except Exception as e:
            print(f"⚠️ Không thể đăng ký hook keyboard: {e}")

    tasks = [
        asyncio.create_task(vision_worker()),
        asyncio.create_task(llm_worker()),
        asyncio.create_task(tts_worker())
    ]
    yield
    for task in tasks:
        task.cancel()
    if KEYBOARD_AVAILABLE:
        try:
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass

app = FastAPI(title="VSL Realtime Pipeline", lifespan=lifespan)

@app.websocket("/ws/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"🔌 [WebSocket] Client đã kết nối!")
    try:
        while True:
            # Nhận cả bytes (video frame) hoặc text (lệnh điều khiển từ Web Client)
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                if frame_queue.full():
                    try:
                        _ = frame_queue.get_nowait() 
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put(data)
                
            elif "text" in message and message["text"]:
                text_msg = message["text"].strip().lower()
                # Nếu client web bắt phím Space và gửi text qua socket
                if text_msg in ["space", "toggle", "toggle_recording"]:
                    toggle_recording()
                    await websocket.send_json({
                        "type": "status", 
                        "is_recording": IS_RECORDING
                    })
            
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