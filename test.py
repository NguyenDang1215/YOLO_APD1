import pandas as pd
from ultralytics.nn.tasks import DetectionModel

# 1. Load model của bạn
cfg_path = 'ultralytics/cfg/models/v8/yolov8m_goldacsim_edge.yaml'
model = DetectionModel(cfg_path)

# 2. Khai báo danh sách chứa dữ liệu
layers_data = []

# 3. Quét qua từng lớp trong mô hình
for i, m in enumerate(model.model):
    name = m._get_name()
    # Tính tổng số tham số của lớp
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    # Lấy thông tin đầu ra (nếu có)
    # Ghi nhận lại cấu trúc để tạo bảng
    layers_data.append({
        "ID": i,
        "Module_Name": name,
        "Parameters": params
    })

# 4. Chuyển thành DataFrame và lưu ra file CSV
df = pd.DataFrame(layers_data)

# Tính thêm phần trăm tỷ trọng của từng lớp
total_params = df['Parameters'].sum()
df['Percentage (%)'] = (df['Parameters'] / total_params * 100).round(2)

# Xuất file
df.to_csv('Model_Architecture_Stats.csv', index=False)
print("Đã xuất báo cáo thành công ra file: Model_Architecture_Stats.csv")