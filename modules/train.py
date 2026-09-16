print("🚀 [1] Đang nạp thư viện PyTorch...", flush=True)

import os
import csv
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from thop import profile

from dataset import VSLDataset
from model import create_model  # ĐÃ THAY ĐỔI: Dùng Model Factory thay vì gọi cứng VSLModel

print("✅ [2] Đã nạp xong thư viện! Bắt đầu chạy code...", flush=True)

# ==========================================
# CÁC HÀM HỖ TRỢ & CẤU HÌNH
# ==========================================
def load_config(config_path=r"C:/Users/dotru/STUDIE/Competition/Sang_tao_tre_AI/VSL_pipeline/configs/config.yaml"): # Nên dùng đường dẫn tương đối để dễ di chuyển code
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    """Khóa toàn bộ sự ngẫu nhiên để đảm bảo tính lặp lại"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Đã khóa Seed ở mức {seed}. Kết quả huấn luyện sẽ nhất quán 100%!")

def count_parameters(model):
    """Đếm tổng số tham số (parameters) của mô hình"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model, device, cfg):
    """Đo lường chi phí phần cứng và tốc độ thực thi của Model"""
    model.eval()
    
    in_channels = cfg['model'].get('in_channels', 9)
    seq_len = cfg['data'].get('sequence_length', 48)
    num_nodes = cfg['model'].get('num_nodes', 76)
    
    # Tạo dữ liệu giả lập
    dummy_input = torch.randn(1, in_channels, seq_len, num_nodes).to(device)
    
    # Đo FLOPS và Tổng Parameters bằng thop
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    # Đo VRAM/RAM (Parameter Memory)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    # Đo tốc độ Inference (Warm-up GPU trước)
    with torch.no_grad():
        for _ in range(10): 
            _ = model(dummy_input)
            
    iterations = 100
    times = []
    
    with torch.no_grad():
        if device.type == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(iterations):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
        else:
            for _ in range(iterations):
                start = time.time()
                _ = model(dummy_input)
                times.append((time.time() - start) * 1000)
                
    avg_infer_time = sum(times) / iterations
    fps = 1000 / avg_infer_time if avg_infer_time > 0 else 0
    
    model.train()
    return flops, params, param_memory, avg_infer_time, fps

# ==========================================
# CÁC HÀM HUẤN LUYỆN
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for seqs, labels in pbar:
        seqs, labels = seqs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(seqs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # Gỡ scale của AMP ra trước
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"Loss": running_loss / (pbar.n + 1), "Acc": correct / total})

    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(dataloader), correct / total

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    set_seed(42)
    cfg = load_config()

    # 1. TẠO ĐƯỜNG DẪN LƯU TRỮ ĐỘNG CHỐNG GHI ĐÈ
    model_name = cfg['experiment']['model_name']
    run_name = cfg['experiment']['run_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    save_dir = os.path.join(cfg['train']['checkpoint_dir'], model_name, f"{run_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Mọi kết quả của lần chạy này sẽ được lưu an toàn tại: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}", flush=True)
    
    # 2. KHỞI TẠO DỮ LIỆU
    train_dataset = VSLDataset(cfg['data']['base_dir'], cfg['data']['json_meta'], split="train", sequence_length=cfg['data']['sequence_length'])
    val_dataset = VSLDataset(cfg['data']['base_dir'], cfg['data']['json_meta'], split="test", sequence_length=cfg['data']['sequence_length'])
    
    # Xuất từ điển nhãn
    label_map_path = "label_map_472.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset.label_map, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã xuất từ điển ra file {label_map_path}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, 
        num_workers=cfg['data']['num_workers'], pin_memory=True, 
        persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, 
        num_workers=cfg['data']['num_workers'], pin_memory=True, persistent_workers=True
    )
        
    # 3. KHỞI TẠO MODEL QUA FACTORY
    model = create_model(cfg, train_dataset.num_classes).to(device)
    
    # 4. BENCHMARK PHẦN CỨNG & XUẤT SUMMARY
    print("🔍 Đang phân tích phần cứng (Memory, FLOPS, Inference Speed)...")
    flops, params, param_memory, avg_infer_time, fps = benchmark_model(model, device, cfg)
    
    with open(os.path.join(save_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== KẾT QUẢ BENCHMARK MÔ HÌNH ===\n")
        f.write(f"Model Name       : {model_name.upper()}\n")
        f.write(f"Run Name         : {run_name}\n")
        f.write(f"---------------------------------\n")
        f.write(f"Parameters       : {params / 1e6:.3f} M (Triệu tham số)\n")
        f.write(f"Param Memory     : {param_memory:.2f} MB\n")
        f.write(f"MACs / FLOPS     : {flops / 1e9:.3f} G (Tỷ phép tính / sequence)\n")
        f.write(f"Inference Time   : {avg_infer_time:.2f} ms / sequence\n")
        f.write(f"Max Throughput   : {fps:.0f} FPS (Khung hình/giây)\n")
        f.write(f"=================================\n\n")
        f.write(str(model))
        
    print(f"📊 Report: {params/1e6:.2f}M Params | {param_memory:.2f} MB | {flops/1e9:.2f}G FLOPS | Speed: {fps:.0f} FPS")

    # 5. CẤU HÌNH LOGGING (TENSORBOARD & CSV)
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
    csv_path = os.path.join(save_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "LR", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])

    # 6. OPTIMIZER, LOSS & SCHEDULER
    t_cfg = cfg['train'] # Trỏ tới block 'train' trong config
    
    # Đọc thông số Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg.get('label_smoothing', 0.1))
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=t_cfg.get('learning_rate', 0.001), 
        weight_decay=t_cfg.get('weight_decay', 0.01)
    )
    scaler = torch.amp.GradScaler('cuda')
    
    # Đọc thông số Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=t_cfg.get('scheduler_factor', 0.5), 
        patience=t_cfg.get('scheduler_patience', 4), 
        min_lr=float(t_cfg.get('min_lr', 1e-6)), 
    )
    
    best_val_acc = 0.0
    
    # 7. VÒNG LẶP HUẤN LUYỆN
    for epoch in range(1, cfg['train']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Ghi Log vào CSV
        csv_writer.writerow([epoch, current_lr, train_loss, train_acc, val_loss, val_acc])
        csv_file.flush()
        
        # Ghi Log vào TensorBoard
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        tb_writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Lưu Checkpoint tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, "best_vsl_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, save_path)
            print(f"--> ⭐ Đã lưu Checkpoint mới! (Val Acc: {best_val_acc:.4f})")

    # 8. DỌN DẸP & LƯU KẾT QUẢ CUỐI
    csv_file.close()
    tb_writer.close()
    print(f"✅ Đã hoàn thành huấn luyện! Best Val Acc: {best_val_acc:.4f}")
    
    with open(os.path.join(save_dir, "model_summary.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n=== KẾT QUẢ HUẤN LUYỆN ===\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")

if __name__ == "__main__":
    main()