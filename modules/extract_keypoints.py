import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import shutil
import uuid
import multiprocessing as mproc
from core_vision import extract_standard_landmarks

# ⚠️ ĐƯỜNG DẪN CỦA BẠN
VIDEO_DIR = r"C:\Users\dotru\STUDIE\Competition\Sang_tao_tre_AI\processed_augmented\frame_splited"
OUTPUT_DIR = r"C:\Users\dotru\STUDIE\Competition\Sang_tao_tre_AI\processed_augmented\keypoints_v2"

# Biến toàn cục ĐỂ TỪNG CORE (Nhân CPU) TỰ GIỮ 1 BẢN MEDIAPIPE RIÊNG
global_holistic = None

def init_worker():
    """Hàm này chạy 1 lần duy nhất trên MỖI NHÂN CPU khi khởi động worker"""
    global global_holistic
    global_holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

def process_single_video(args):
    """Hàm này chỉ lo xử lý đúng 1 video, sẽ được các nhân CPU tranh nhau làm"""
    split, gloss, vid = args
    video_path = os.path.join(VIDEO_DIR, split, gloss, vid)
    out_folder = os.path.join(OUTPUT_DIR, split, gloss)
    out_path = os.path.join(out_folder, vid.replace('.mp4', '.npy'))
    
    # 1. Bỏ qua nếu đã chạy rồi (Hỗ trợ Resume cực nhanh)
    if os.path.exists(out_path):
        return True
        
    # 2. Tạo tên file tạm UNIQUE (Tránh việc các nhân CPU dẫm đạp lên nhau)
    temp_video_path = f"temp_{uuid.uuid4().hex}.mp4"
    
    try:
        # 3. Copy file tạm để né lỗi Unicode tiếng Việt của OpenCV
        shutil.copy2(video_path, temp_video_path)
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            if os.path.exists(temp_video_path): os.remove(temp_video_path)
            return False

        frames_landmarks = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # 4. Gọi bản MediaPipe đã được nạp sẵn trên RAM của nhân CPU này
            results = global_holistic.process(frame_rgb)
            
            # Trích xuất chuẩn (Dùng hàm trong core_vision.py)
            landmarks = extract_standard_landmarks(results)
            frames_landmarks.append(landmarks)
            
        cap.release()
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        
        if len(frames_landmarks) > 0:
            os.makedirs(out_folder, exist_ok=True)
            np.save(out_path, np.array(frames_landmarks))
            return True
        return False

    except Exception as e:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_videos = []
    already_done_count = 0 # Biến đếm số video đã trích xuất
    
    print("⏳ Đang quét kiểm tra tiến độ cũ (Tính năng Resume)...")
    for split in ['train', 'test']:
        split_dir = os.path.join(VIDEO_DIR, split)
        if not os.path.exists(split_dir): continue
        for gloss in os.listdir(split_dir):
            gloss_dir = os.path.join(split_dir, gloss)
            if not os.path.isdir(gloss_dir): continue
            for vid in os.listdir(gloss_dir):
                if vid.endswith('.mp4'):
                    # --- RESUME THÔNG MINH: Kiểm tra file đích trước khi thêm vào danh sách ---
                    out_folder = os.path.join(OUTPUT_DIR, split, gloss)
                    out_path = os.path.join(out_folder, vid.replace('.mp4', '.npy'))
                    
                    if os.path.exists(out_path):
                        already_done_count += 1
                    else:
                        all_videos.append((split, gloss, vid))
                        
    total_videos = len(all_videos) + already_done_count
    print(f"🔍 Tổng tài nguyên: {total_videos} videos.")
    
    if already_done_count > 0:
        print(f"⏩ [RESUME] Đã bỏ qua {already_done_count} video xử lý từ trước.")
        
    print(f"⏳ Nhiệm vụ còn lại: {len(all_videos)} videos.")
    
    if len(all_videos) == 0:
        print("✅ Toàn bộ dữ liệu đã được trích xuất xong. Bạn có thể mang đi Train!")
        return
    
    # BẬT ĐA LUỒNG (Đa tiến trình)
    num_cores = max(1, mproc.cpu_count() - 2) 
    print(f"🚀 Kích hoạt ĐỘNG CƠ {num_cores} NHÂN CPU. Chuẩn bị cày...")
    
    with mproc.Pool(processes=num_cores, initializer=init_worker) as pool:
        list(tqdm(pool.imap_unordered(process_single_video, all_videos), total=len(all_videos), desc="🔥 Tiến độ"))

if __name__ == "__main__":
    mproc.freeze_support()
    main()