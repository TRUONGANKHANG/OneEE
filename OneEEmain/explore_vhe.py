import json
from sklearn.model_selection import train_test_split

# 1. Đọc toàn bộ dữ liệu
with open("vhe-dataset-main/event-extraction/event.json", "r", encoding="utf-8") as f:
    data = json.load(f)      # data là 1 list các mẫu

print("Tổng số mẫu:", len(data))

# 2. Chia train (70%) và phần còn lại (30%)
train_data, temp_data = train_test_split(
    data,
    test_size=0.30,
    random_state=42,
    shuffle=True,
)

# 3. Chia phần còn lại thành dev (15%) và test (15%)
dev_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,           # vì 0.5 * 30% = 15%
    random_state=42,
    shuffle=True,
)

print("Train:", len(train_data))
print("Dev  :", len(dev_data))
print("Test :", len(test_data))

# 4. Ghi ra file theo đúng format gốc
with open("vhe-dataset-main/event-extraction/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("vhe-dataset-main/event-extraction/dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_data, f, ensure_ascii=False, indent=2)

with open("vhe-dataset-main/event-extraction/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
