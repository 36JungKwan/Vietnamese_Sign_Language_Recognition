import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class VSLDataset(Dataset):
    def __init__(self, data_dir, json_metadata_path, split="train", sequence_length=48):
        """
        data_dir: Đường dẫn tới thư mục `keypoints_splited`
        json_metadata_path: Đường dẫn tới file `vsl_full_front_augmented.json`
        split: "train" hoặc "test"
        """
        self.sequence_length = sequence_length
        self.split = split
        self.samples = []
        
        # 1. Đọc file JSON Metadata
        with open(json_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # 2. Tự động gom toàn bộ nhãn (gloss) duy nhất để làm từ điển
        unique_glosses = sorted(list(set([item['gloss'] for item in metadata])))
        self.label_map = {gloss: idx for idx, gloss in enumerate(unique_glosses)}
        self.num_classes = len(self.label_map)
        
        # 3. Lọc file theo Split (train hoặc test) và lưu đường dẫn
        missing_files = 0
        for item in metadata:
            if item['split'] == split:
                gloss = item['gloss']
                vid = item['videoid']
                
                # Cấu trúc: keypoints_splited / train / An ủi / 127824.npy
                file_path = os.path.join(data_dir, split, gloss, f"{vid}.npy")
                
                if os.path.exists(file_path):
                    self.samples.append({
                        'path': file_path,
                        'label': self.label_map[gloss]
                    })
                else:
                    missing_files += 1
                    
        print(f"📊 [Tập {split.upper()}] Nạp thành công {len(self.samples)} video. (Thiếu {missing_files} file).")

    def __len__(self):
        return len(self.samples)

    def interpolate_sequence(self, data):
        """Ép video có độ dài T (vd: 30, 55...) về chuẩn 48 frames"""
        T, V, C = data.shape
        if T == self.sequence_length:
            return data

        # Chuyển (T, 76, 3) -> (1, 228, T) để nội suy 1D
        data_tensor = torch.tensor(data, dtype=torch.float32).reshape(T, V * C).permute(1, 0).unsqueeze(0)
        
        # Co/giãn tuyến tính theo thời gian
        interpolated = F.interpolate(data_tensor, size=self.sequence_length, mode='linear', align_corners=False)
        
        # Phục hồi hình dáng -> (48, 76, 3)
        return interpolated.squeeze(0).permute(1, 0).reshape(self.sequence_length, V, C).numpy()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample['path'])  

        # 1. Ép số frame về chuẩn (vd: 48)
        data = self.interpolate_sequence(data)

        # 2. CHUẨN HÓA GỐC TỌA ĐỘ (Hệ quy chiếu Cổ - Node 75)
        # Ép điểm cổ của mọi frame về tọa độ (0, 0, 0)
        neck_coords = data[:, 75:76, :] 
        data = data - neck_coords       

        # 3. Tính Vận tốc (Đạo hàm bậc 1)
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        velocity[0] = velocity[1] 

        # 4. Tính Gia tốc (Đạo hàm bậc 2)
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = acceleration[1]

        # 5. Gộp 9 kênh (x,y,z, vx,vy,vz, ax,ay,az) -> Shape: (48, 76, 9)
        combined = np.concatenate([data, velocity, acceleration], axis=-1)

        # 6. Trả về đúng chiều Model cần: (Channels, Time, Nodes) -> (9, 48, 76)
        tensor = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return tensor, label