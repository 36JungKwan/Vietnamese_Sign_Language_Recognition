import numpy as np
import torch

# Số lượng node chuẩn
NUM_NODES = 76

def extract_standard_landmarks(results):
    """
    Trích xuất chuẩn 76 điểm (x, y, z) theo đúng thứ tự:
    1. Pose (33 điểm, index 0-32)
    2. Left Hand (21 điểm, index 33-53)
    3. Right Hand (21 điểm, index 54-74)
    4. Neck (1 điểm, index 75)
    """
    landmarks = np.zeros((NUM_NODES, 3), dtype=np.float32)
    
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
            
    # 2. Left Hand
    if results.left_hand_landmarks:
        for i in range(21):
            lm = results.left_hand_landmarks.landmark[i]
            landmarks[33 + i] = [lm.x, lm.y, lm.z]
            
    # 3. Right Hand
    if results.right_hand_landmarks:
        for i in range(21):
            lm = results.right_hand_landmarks.landmark[i]
            landmarks[54 + i] = [lm.x, lm.y, lm.z]
            
    return landmarks

def normalize_scale_and_coords(data):
    """
    Chuẩn hóa Tọa độ (Ép cổ về gốc 0,0,0) và Kích thước (Chia cho độ rộng vai).
    Đầu vào/Đầu ra: data có shape (T, 76, 3)
    """
    # 1. Ép cổ về (0,0,0)
    neck_coords = data[:, 75:76, :] 
    data = data - neck_coords
    
    # 2. Chuẩn hóa tỷ lệ kích thước cơ thể (Scale Normalization)
    left_shoulder = data[:, 11:12, :]
    right_shoulder = data[:, 12:13, :]
    
    # Tính khoảng cách Euclidean giữa 2 vai
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
    # Ngăn chia cho 0 nếu MediaPipe lỗi không bắt được vai
    shoulder_width = np.where(shoulder_width < 1e-5, 1.0, shoulder_width)
    
    # Chia toàn bộ tọa độ cho độ rộng vai
    data = data / shoulder_width
    
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