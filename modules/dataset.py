import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

# Kéo "Trái tim chuẩn hóa" vào
from core_vision import normalize_scale_and_coords, build_9_channel_tensor

class VSLDataset(Dataset):
    def __init__(self, data_dir, json_metadata_path, split="train", sequence_length=48):
        self.sequence_length = sequence_length
        self.split = split
        self.samples = []
        
        # 1. Đọc file JSON Metadata
        with open(json_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # 2. Tự động gom toàn bộ nhãn (gloss)
        unique_glosses = sorted(list(set([item['gloss'] for item in metadata])))
        self.label_map = {gloss: idx for idx, gloss in enumerate(unique_glosses)}
        
        # --- TÍNH NĂNG MỚI: Thêm class IDLE (Đứng yên) ---
        self.label_map['Idle'] = len(self.label_map) # Sẽ chiếm ID 472
        self.num_classes = len(self.label_map)
        
        # 3. Lọc file theo Split (train hoặc test)
        missing_files = 0
        for item in metadata:
            if item['split'] == split:
                gloss = item['gloss']
                vid = item['videoid']
                file_path = os.path.join(data_dir, split, gloss, f"{vid}.npy")
                
                if os.path.exists(file_path):
                    self.samples.append({
                        'path': file_path,
                        'label': self.label_map[gloss]
                    })
                else:
                    missing_files += 1
                    
        # --- TÍNH NĂNG MỚI: Bơm thêm 10% dữ liệu là trạng thái Đứng Yên (Idle) ---
        self.idle_count = int(len(self.samples) * 0.1)
        self.total_length = len(self.samples) + self.idle_count
                    
        print(f"📊 [Tập {split.upper()}] Nạp {len(self.samples)} video thật. Sinh thêm {self.idle_count} video IDLE. (Thiếu {missing_files} file).")

    def __len__(self):
        return self.total_length

    def interpolate_sequence(self, data):
        T, V, C = data.shape
        if T == self.sequence_length:
            return data
        data_tensor = torch.tensor(data, dtype=torch.float32).reshape(T, V * C).permute(1, 0).unsqueeze(0)
        interpolated = F.interpolate(data_tensor, size=self.sequence_length, mode='linear', align_corners=False)
        return interpolated.squeeze(0).permute(1, 0).reshape(self.sequence_length, V, C).numpy()

    def __getitem__(self, idx):
        # NẾU RƠI VÀO VÙNG DỮ LIỆU IDLE (Tự động sinh mảng đứng yên)
        if idx >= len(self.samples):
            # Lấy ngẫu nhiên 1 video thật
            real_sample = self.samples[np.random.randint(0, len(self.samples))]
            data = np.load(real_sample['path'])
            # Lấy đúng frame đầu tiên (lúc người múa đang hạ tay chuẩn bị)
            first_frame = data[0:1] 
            # Nhân bản lên 48 frames + Thêm nhiễu siêu nhỏ giả lập rung tay do nhịp tim
            idle_data = np.repeat(first_frame, self.sequence_length, axis=0)
            idle_data += np.random.normal(0, 0.001, idle_data.shape)
            
            data_norm = normalize_scale_and_coords(idle_data)
            tensor = build_9_channel_tensor(data_norm)
            label = torch.tensor(self.label_map['Idle'], dtype=torch.long)
            return tensor, label

        # NẾU LÀ DỮ LIỆU THẬT MÚA KÝ HIỆU
        sample = self.samples[idx]
        data = np.load(sample['path'])  

        # 1. Ép frame về 48
        data = self.interpolate_sequence(data)

        # 2. CHUẨN HÓA (Dùng hàm chung 100% với App.py)
        data_norm = normalize_scale_and_coords(data)
        
        # 3. XÂY DỰNG TENSOR 9 KÊNH (Dùng hàm chung 100% với App.py)
        tensor = build_9_channel_tensor(data_norm)
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        return tensor, label