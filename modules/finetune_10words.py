import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CTR_GCN_Model
from dataset import VSLDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🔒 Đã khóa Random Seed = {seed}")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Thiết bị sử dụng: {device}")

    # Đường dẫn
    base_dir = "C:/Users/dotru/STUDIE/Competition/Sang_tao_tre_AI/processed_augmented/keypoints_v2"
    meta_path = "configs/subsets/metadata_10_words.json"
    full_label_path = "modules/label_map_full_472.json"
    target_label_path = "modules/label_map_472_10w.json"
    pretrained_ckpt_path = "modules/checkpoints/ctr_gcn/baseline_20260917_1148/best_vsl_model.pth"
    save_dir = "modules/checkpoints/ctr_gcn/finetuned_10words"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Tải từ điển nhãn
    with open(full_label_path, "r", encoding="utf-8") as f:
        full_map = json.load(f)
    with open(target_label_path, "r", encoding="utf-8") as f:
        target_map = json.load(f)

    id2word = {idx: word for word, idx in target_map.items()}
    num_classes = len(target_map)
    print(f"🎯 Số lượng từ mục tiêu: {num_classes}")
    for w, idx in target_map.items():
        print(f"   [{idx}] {w} (idx gốc: {full_map[w]})")

    # 2. Khởi tạo Datasets & DataLoaders
    train_dataset = VSLDataset(base_dir, meta_path, split="train", sequence_length=48)
    val_dataset = VSLDataset(base_dir, meta_path, split="test", sequence_length=48)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=(device.type == "cuda")
    )

    # 3. Khởi tạo mô hình & Nạp Pre-trained Checkpoint
    print(f"\n📦 Đang nạp Pre-trained Backbone từ: {pretrained_ckpt_path}")
    base_model = CTR_GCN_Model(num_classes=473, in_channels=9, num_nodes=76, d_model=128)
    ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
    # Lọc bỏ buffer đo benchmark thop (total_ops, total_params)
    clean_sd = {k: v for k, v in ckpt["model_state_dict"].items() if "total_ops" not in k and "total_params" not in k}
    base_model.load_state_dict(clean_sd, strict=True)
    print("✅ Đã nạp thành công trọng số pretrained của mô hình 473 từ!")

    # 4. Chuyển giao trọng số Head Classifier (Weight Surgery)
    old_indices = [full_map[w] for w in target_map.keys()]
    old_weight = base_model.classifier.weight.data
    old_bias = base_model.classifier.bias.data

    new_classifier = nn.Linear(128, num_classes)
    with torch.no_grad():
        new_classifier.weight.copy_(old_weight[old_indices, :])
        new_classifier.bias.copy_(old_bias[old_indices])
    base_model.classifier = new_classifier
    print(f"✅ Đã phẫu thuật classifier: shape mới = {base_model.classifier.weight.shape}")

    model = base_model.to(device)

    # 5. Optimizer với Differential Learning Rate & Loss Function
    # Backbone: lr nhỏ để giữ trọn vẹn feature representation không bị lãng quên (catastrophic forgetting)
    # Classifier Head: lr cao hơn một chút để hội tụ nhanh với không gian 10 nhãn
    optimizer = optim.AdamW([
        {"params": model.ctr_gcn.parameters(), "lr": 5e-5, "weight_decay": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 1e-4}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 6. Đánh giá Zero-shot trước khi Fine-tune
    model.eval()
    init_correct = 0
    init_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            init_correct += (preds == y).sum().item()
            init_total += y.size(0)
    print(f"\n📊 Độ chính xác Zero-shot (trước fine-tune): {init_correct}/{init_total} = {init_correct/init_total*100:.2f}%\n")

    # 7. Fine-tuning Loop (15 Epochs)
    epochs = 15
    best_val_acc = 0.0
    best_save_path = os.path.join(save_dir, "best_vsl_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for seqs, labels in pbar:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            pbar.set_postfix({
                "Loss": f"{running_loss / (pbar.n + 1):.4f}", 
                "Acc": f"{train_correct / train_total:.4f}"
            })

        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                for p, t in zip(preds, labels):
                    class_total[t.item()] += 1
                    if p == t:
                        class_correct[t.item()] += 1

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        scheduler.step()

        bb_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(f"👉 Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR (bb/head): {bb_lr:.2e}/{head_lr:.2e}")

        # Lưu Checkpoint tốt nhất
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "label_map": target_map
            }, best_save_path)
            print(f"   ⭐ Đã lưu Best Checkpoint mới! (Val Acc: {val_acc*100:.2f}%)")

    # 8. Đánh giá chi tiết mô hình tốt nhất
    print("\n" + "=" * 60)
    print(f"🏆 Huấn luyện hoàn tất! Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"💾 Checkpoint lưu tại: {best_save_path}")
    
    # Load best checkpoint và in chi tiết từng lớp
    best_ckpt = torch.load(best_save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    c_corr = [0] * num_classes
    c_tot = [0] * num_classes
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            preds = outputs.argmax(dim=1)
            for p, t in zip(preds, labels):
                c_tot[t.item()] += 1
                if p == t:
                    c_corr[t.item()] += 1

    print("\n📊 BẢNG ĐỘ CHÍNH XÁC CHI TIẾT TỪNG TỪ:")
    print(f"{'STT':<4} {'Từ vựng':<18} {'Độ chính xác':<15} {'Số mẫu':<10}")
    print("-" * 50)
    for idx in range(num_classes):
        word = id2word[idx]
        acc_str = f"{c_corr[idx]}/{c_tot[idx]} ({c_corr[idx]/max(1, c_tot[idx])*100:.1f}%)"
        print(f"{idx:<4} {word:<18} {acc_str:<15} {c_tot[idx]:<10}")
    print("=" * 60)

    # Lưu label_map tương ứng vào thư mục checkpoint
    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(target_map, f, ensure_ascii=False, indent=4)
    print("✅ Đã lưu file nhãn đi kèm tại checkpoint folder.")

if __name__ == "__main__":
    main()
