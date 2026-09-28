import numpy as np
import torch

# Số lượng node chuẩn
NUM_NODES = 76

def extract_standard_landmarks(results, prev_landmarks=None, hand_miss_counters=None):
    """
    Trích xuất chuẩn 76 điểm (x, y, z) theo đúng thứ tự:
    1. Pose (33 điểm, index 0-32)
    2. Left Hand (21 điểm, index 33-53)
    3. Right Hand (21 điểm, index 54-74)
    4. Neck (1 điểm, index 75)
    
    Hỗ trợ cơ chế Hold Last Valid Position: nếu mất dấu tay do motion-blur trong 1-6 frames,
    giữ nguyên vị trí hợp lệ gần nhất để triệt tiêu xung vận tốc/gia tốc ảo (Velocity Spikes).
    """
    landmarks = np.zeros((NUM_NODES, 3), dtype=np.float32)
    
    if hand_miss_counters is None:
        hand_miss_counters = {"left": 0, "right": 0}
    
    # 1. Pose
    if results.pose_landmarks:
        for i in range(33):
            lm = results.pose_landmarks.landmark[i]
            landmarks[i] = [lm.x, lm.y, lm.z]
            
        # 4. Tính điểm Cổ (Neck) - Trung điểm của 2 vai (11 và 12)
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        landmarks[75] = [
            (l_sh.x + r_sh.x) / 2.0,
            (l_sh.y + r_sh.y) / 2.0,
            (l_sh.z + r_sh.z) / 2.0
        ]
    elif prev_landmarks is not None:
        landmarks[:33] = prev_landmarks[:33]
        landmarks[75] = prev_landmarks[75]
            
    # 2. Left Hand
    if results.left_hand_landmarks:
        hand_miss_counters["left"] = 0
        for i in range(21):
            lm = results.left_hand_landmarks.landmark[i]
            landmarks[33 + i] = [lm.x, lm.y, lm.z]
    elif prev_landmarks is not None and hand_miss_counters["left"] < 8:
        # Giữ lại vị trí hợp lệ gần nhất trong tối đa 8 frames (khoảng 0.4s mất dấu tạm thời)
        landmarks[33:54] = prev_landmarks[33:54]
        hand_miss_counters["left"] += 1
    else:
        hand_miss_counters["left"] += 1
            
    # 3. Right Hand
    if results.right_hand_landmarks:
        hand_miss_counters["right"] = 0
        for i in range(21):
            lm = results.right_hand_landmarks.landmark[i]
            landmarks[54 + i] = [lm.x, lm.y, lm.z]
    elif prev_landmarks is not None and hand_miss_counters["right"] < 8:
        # Giữ lại vị trí hợp lệ gần nhất trong tối đa 8 frames
        landmarks[54:75] = prev_landmarks[54:75]
        hand_miss_counters["right"] += 1
    else:
        hand_miss_counters["right"] += 1
            
    return landmarks

def normalize_scale_and_coords(data):
    """
    Chuẩn hóa Tọa độ (Ép cổ về gốc 0,0,0) và Kích thước (Chia cho độ rộng vai).
    Đầu vào/Đầu ra: data có shape (T, 76, 3)
    
    Có Fallback Scale thông minh: nếu vai bị khuất, cắt mép khung hình hoặc người ngồi quá gần
    (shoulder_width < 0.08), tự động dùng tỷ lệ khoảng cách Cổ - Mũi để giữ tỷ lệ toàn thân.
    """
    # 1. Ép cổ về (0,0,0)
    neck_coords = data[:, 75:76, :] 
    data = data - neck_coords
    
    # 2. Chuẩn hóa tỷ lệ kích thước cơ thể (Scale Normalization)
    left_shoulder = data[:, 11:12, :]
    right_shoulder = data[:, 12:13, :]
    
    # Tính khoảng cách Euclidean giữa 2 vai
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
    
    # Fallback scale từ Cổ tới Mũi (Nose index 0)
    nose = data[:, 0:1, :]
    neck = data[:, 75:76, :]  # (0, 0, 0)
    neck_to_nose = np.linalg.norm(neck - nose, axis=-1, keepdims=True)
    fallback_scale = np.where(neck_to_nose > 0.04, neck_to_nose * 1.8, 0.3)
    
    scale = np.where(shoulder_width > 0.08, shoulder_width, fallback_scale)
    scale = np.where(scale < 1e-4, 0.3, scale)
    
    # Chia toàn bộ tọa độ cho scale
    data = data / scale
    
    return data

def build_9_channel_tensor(data):
    """
    Tính Vận tốc, Gia tốc và gộp thành 9 kênh.
    Đầu vào: data đã được chuẩn hóa (T, 76, 3)
    Đầu ra: Tensor shape (9, T, 76) để đưa thẳng vào Model
    """
    velocity = np.zeros_like(data)
    velocity[1:] = data[1:] - data[:-1]
    velocity[0] = velocity[1] 
    
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]
    
    combined = np.concatenate([data, velocity, acceleration], axis=-1) # (T, 76, 9)
    tensor = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1) # (9, T, 76)
    
    return tensor