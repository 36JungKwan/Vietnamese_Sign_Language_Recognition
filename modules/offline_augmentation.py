import os
import numpy as np
import multiprocessing as mproc
from tqdm import tqdm
from scipy.ndimage import zoom

DATA_DIR = r"C:\Users\dotru\STUDIE\Competition\Sang_tao_tre_AI\processed_augmented\keypoints_v2"
# Số lượng biến thể sinh thêm cho mỗi file gốc (Ví dụ: 2 -> Tổng data x3)
AUGMENT_MULTIPLIER = 2 

def augment_scale_and_shift(data):
    """Phóng to/thu nhỏ 0.9x - 1.1x và dịch chuyển nhẹ tâm"""
    scale_factor = np.random.uniform(0.9, 1.1)
    shift_factor = np.random.uniform(-0.05, 0.05, size=(1, 1, 3))
    return (data * scale_factor) + shift_factor

def augment_rotate_z(data):
    """Xoay toàn bộ khung xương quanh trục Z (Góc nhìn camera) +- 15 độ"""
    angle = np.random.uniform(-15, 15)
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)
    
    # data shape: (T, V, 3). Nhân ma trận với chiều tọa độ
    return np.dot(data, rotation_matrix.T)

def augment_temporal_warp(data, target_frames=48):
    """Co giãn thời gian (nhanh/chậm lên) rồi ép lại 48 frames"""
    speed_factor = np.random.uniform(0.8, 1.25)
    T, V, C = data.shape
    new_T = max(int(T * speed_factor), 15)
    
    # Nội suy theo chiều thời gian
    warped_data = np.zeros((new_T, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            warped_data[:, v, c] = zoom(data[:, v, c], new_T / T, mode='nearest')
            
    # Ép ngược về target_frames (để cùng chuẩn đầu vào)
    final_data = np.zeros((target_frames, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            final_data[:, v, c] = zoom(warped_data[:, v, c], target_frames / new_T, mode='nearest')
    return final_data

def process_single_file(file_path):
    # Bỏ qua nếu file này đã là file augment để tránh lặp vô hạn
    if "_aug_" in file_path:
        return
        
    try:
        data = np.load(file_path)
    except:
        return # Bỏ qua file hỏng

    for i in range(AUGMENT_MULTIPLIER):
        aug_path = file_path.replace(".npy", f"_aug_{i}.npy")
        if os.path.exists(aug_path):
            continue
            
        # Trộn các hiệu ứng ngẫu nhiên
        aug_data = data.copy()
        
        # 50% xác suất xoay, 50% xác suất scale
        if np.random.rand() > 0.5:
            aug_data = augment_rotate_z(aug_data)
        else:
            aug_data = augment_scale_and_shift(aug_data)
            
        # 50% xác suất bóp méo thời gian
        if np.random.rand() > 0.5:
            aug_data = augment_temporal_warp(aug_data)
            
        np.save(aug_path, aug_data)

def main():
    npy_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.npy') and "_aug_" not in file:
                npy_files.append(os.path.join(root, file))
                
    print(f"🔍 Tìm thấy {len(npy_files)} video gốc. Sẽ đẻ thêm {len(npy_files) * AUGMENT_MULTIPLIER} videos...")
    
    num_cores = max(1, mproc.cpu_count() - 1)
    with mproc.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_single_file, npy_files), total=len(npy_files), desc="🛠 Augmenting"))

if __name__ == "__main__":
    mproc.freeze_support()
    main()