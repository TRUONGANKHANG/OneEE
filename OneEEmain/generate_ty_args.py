# 📄 File: generate_ty_args.py
import json
import os
from collections import defaultdict

DATA_PATH = "./data/bkee/train.json"
OUTPUT_PATH = "./data/bkee/ty_args.json"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Không tìm thấy file {DATA_PATH}")
        return

    data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Lỗi đọc dòng JSON: {e}")
                continue

    mapping = defaultdict(set)

    for doc in data:
        for event in doc.get("event_mentions", []):
            e_type = event["event_type"]
            for arg in event.get("arguments", []):
                mapping[e_type].add(arg["role"])

    mapping = {k: sorted(list(v)) for k, v in mapping.items()}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"✅ File {OUTPUT_PATH} đã được tạo thành công với {len(mapping)} loại sự kiện.")
    print("Ví dụ vài dòng đầu:\n")
    for i, (k, v) in enumerate(mapping.items()):
        print(f"  {k}: {v}")
        if i >= 5:
            break

if __name__ == "__main__":
    main()
