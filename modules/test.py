import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VSLDataset
from model import VSLModel

def load_config(config_path="../configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_model():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bắt đầu đánh giá mô hình trên thiết bị: {device}")
    
    # 1. Load tập Test
    test_dataset = VSLDataset(cfg['data']['test_dir'], cfg['data']['label_map'], cfg['data']['sequence_length'])
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    
    # 2. Khởi tạo Model
    model = VSLModel(
        num_classes=test_dataset.num_classes,
        in_channels=cfg['model']['in_channels'],
        d_model=cfg['model']['d_model'],
        num_heads=cfg['model']['num_heads'],
        num_layers=cfg['model']['num_layers']
    ).to(device)
    
    # 3. Load Trọng số (Weights) tốt nhất
    checkpoint_path = os.path.join(cfg['train']['checkpoint_dir'], "best_vsl_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Chưa có file {checkpoint_path}. Bạn cần train xong ít nhất 1 lần để có file này!")
        
    print(f"⏳ Đang nạp trọng số từ {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Nạp thành công! (Mô hình này đạt Val Acc tốt nhất ở Epoch {checkpoint['epoch']})")
    
    # 4. Đánh giá (Evaluation)
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for seqs, labels in pbar:
            seqs, labels = seqs.to(device), labels.to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(seqs)
            
            # Tính Top-1 Accuracy (Đoán trúng phóc 100%)
            _, predicted = outputs.max(1)
            top1_correct += predicted.eq(labels).sum().item()
            
            # Tính Top-5 Accuracy (Đáp án đúng nằm trong top 5 xác suất cao nhất)
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            total += labels.size(0)
            pbar.set_postfix({
                "Top-1 Acc": f"{100.*top1_correct/total:.2f}%",
                "Top-5 Acc": f"{100.*top5_correct/total:.2f}%"
            })
            
    print("\n" + "="*40)
    print("🎯 KẾT QUẢ TEST CUỐI CÙNG:")
    print(f"Tổng số mẫu Test : {total}")
    print(f"Top-1 Accuracy   : {100. * top1_correct / total:.2f}%")
    print(f"Top-5 Accuracy   : {100. * top5_correct / total:.2f}%")
    print("="*40)