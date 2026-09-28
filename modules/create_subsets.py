import json
import random
import os

# ⚠️ ĐIỀN ĐƯỜNG DẪN FILE METADATA GỐC CỦA BẠN (File chứa đầy đủ các video ban đầu)
ORIGINAL_JSON_PATH = r"C:\Users\dotru\STUDIE\Competition\Sang_tao_tre_AI\processed_augmented\vsl_full_front_augmented.json" 
OUTPUT_DIR = "configs/subsets"

# 8 Từ vựng bắt buộc phải có mặt (Không tính Idle vì dataset.py sẽ tự sinh)
CORE_WORDS = ["Tôi", "Khỏe", "Hôm nay", "Bóng chuyền", "Đau", "Cảm ơn", "Chào", "Sinh viên"]

# Các mốc từ vựng bạn muốn tạo (Bao gồm cả Idle)
SUBSET_SIZES = [300, 200, 100, 50, 20, 10]

def main():
    with open(ORIGINAL_JSON_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Lấy danh sách toàn bộ từ vựng có trong JSON
    all_glosses = sorted(list(set([item['gloss'] for item in metadata])))

    # Kiểm tra an toàn
    for w in CORE_WORDS:
        if w not in all_glosses:
            print(f"❌ LỖI: Từ '{w}' không khớp với tên trong metadata gốc! Hãy kiểm tra lại chính tả.")
            return

    # Lọc ra rổ từ vựng ngẫu nhiên (đã trừ đi các từ cốt lõi)
    remaining_glosses = [g for g in all_glosses if g not in CORE_WORDS]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for size in SUBSET_SIZES:
        # Tổng size (VD: 10) = 8 Từ cốt lõi + 1 Idle + (Size - 9) Từ ngẫu nhiên
        num_random_needed = size - len(CORE_WORDS) - 1
        
        if num_random_needed < 0:
            print(f"⚠️ Bỏ qua tập {size} vì không đủ chỗ chứa {len(CORE_WORDS)} từ cốt lõi và Idle.")
            continue

        # Bốc ngẫu nhiên và gộp với nhóm nòng cốt
        selected_random = random.sample(remaining_glosses, num_random_needed)
        selected_glosses = set(CORE_WORDS + selected_random)

        # Lọc lấy các video thuộc nhóm từ vựng đã chọn
        subset_metadata = [item for item in metadata if item['gloss'] in selected_glosses]

        out_path = os.path.join(OUTPUT_DIR, f"metadata_{size}_words.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(subset_metadata, f, ensure_ascii=False, indent=4)

        print(f"✅ Đã tạo tập {size} từ (8 Cốt lõi + {num_random_needed} Ngẫu nhiên + 1 Idle) -> {out_path} ({len(subset_metadata)} videos)")

if __name__ == "__main__":
    # Fix cứng seed để nếu chạy lại nhiều lần, các từ ngẫu nhiên vẫn không đổi
    random.seed(42) 
    main()