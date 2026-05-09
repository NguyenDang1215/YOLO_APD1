import torch
import torch_pruning as tp
from ultralytics import YOLO

# 1. Load mô hình YOLOv8 custom của bạn đã train xong (Teacher)
print("Đang tải mô hình...")
model = YOLO("yolov8n.pt") 
pytorch_model = model.model.eval() # Trích xuất model PyTorch cốt lõi và chuyển sang chế độ đánh giá

# 2. Tạo một input giả (dummy input) để thư viện phân tích cấu trúc đồ thị mạng
# Giả sử ảnh đầu vào của bạn có kích thước 640x640
example_inputs = torch.randn(1, 3, 640, 640)

# 3. Chọn thuật toán đánh giá kênh (Dùng hệ số gamma của lớp Batch Norm)
importance = tp.importance.BNScaleImportance()

# 4. Bảo vệ lớp Đầu ra (Head): CHỈ khóa lớp xuất kết quả cuối cùng
ignored_layers = []
for m in pytorch_model.modules():
    if m.__class__.__name__ == 'Detect':
        # Nhánh cv2 xuất ra tọa độ bounding box
        for seq in m.cv2:
            ignored_layers.append(seq[-1]) # Chỉ khóa lớp Conv2d ngoài cùng
        # Nhánh cv3 xuất ra xác suất các class
        for seq in m.cv3:
            ignored_layers.append(seq[-1]) # Chỉ khóa lớp Conv2d ngoài cùng

# 5. Khởi tạo thuật toán Pruner
# Ở đây ta đặt mục tiêu cắt bỏ 30% số kênh (pruning_ratio = 0.3)
pruner = tp.pruner.MetaPruner(
    pytorch_model,
    example_inputs,
    importance=importance,
    pruning_ratio=0.3, 
    ignored_layers=ignored_layers,
)

# --- THỐNG KÊ TRƯỚC KHI CẮT ---
# Đã sửa thành count_ops_and_params
base_macs, base_params = tp.utils.count_ops_and_params(pytorch_model, example_inputs)
print(f"\n[TRƯỚC KHI CẮT]")
print(f"Tham số (Params): {base_params / 1e6:.2f} Triệu")
print(f"Chi phí tính toán (MACs): {base_macs / 1e9:.2f} G")

# 6. THỰC HIỆN CẮT TỈA (Nhát chém quyết định)
pruner.step()

# --- THỐNG KÊ SAU KHI CẮT ---
# Đã sửa thành count_ops_and_params
pruned_macs, pruned_params = tp.utils.count_ops_and_params(pytorch_model, example_inputs)
print(f"\n[SAU KHI CẮT]")
print(f"Tham số (Params): {pruned_params / 1e6:.2f} Triệu")
print(f"Chi phí tính toán (MACs): {pruned_macs / 1e9:.2f} G")

# 7. Lưu lại mô hình đã cắt
torch.save(pytorch_model, "yolov8_custom_pruned.pt")
print("\nĐã lưu mô hình cắt tỉa thành công!")