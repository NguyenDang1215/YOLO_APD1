from ultralytics import YOLO
import os

def train_model():
    # 1. Khởi tạo mô hình
    # Chúng ta dùng file YAML kiến trúc Edge đã nén, 
    # nhưng nạp thêm trọng số pre-trained của YOLOv8s hoặc m để bắt đầu tốt hơn
    model = YOLO('ultralytics/cfg/models/v8/yolov8m_goldacsim_edge.yaml')
    
    # 2. Cấu hình tham số huấn luyện
    # Lưu ý: Vì nc=1 (chỉ có lớp human), mô hình sẽ hội tụ rất nhanh
    results = model.train(
        data='data/data.yaml',         # Đường dẫn đến file yaml bạn vừa gửi
        epochs=100,               # Số vòng lặp
        imgsz=640,                # Kích thước ảnh
        batch=16,                 # Điều chỉnh tùy theo VRAM của GPU (8, 16, 32)
        device=0,                 # 0 cho GPU, 'cpu' nếu không có card đồ họa
        project='runs/train',     # Nơi lưu kết quả
        name='goldacsim_edge_human', 
        save=True,                # Lưu checkpoint
        plots=True,               # Vẽ biểu đồ loss, mAP
        verbose=True,             # In chi tiết quá trình
        # --- Các kỹ thuật bổ trợ ---
        cos_lr=True,              # Dùng Cosine learning rate scheduler (tốt cho mô hình nhỏ)
        overlap_mask=False,       # Tiết kiệm bộ nhớ
        patience=20               # Dừng sớm nếu model không cải thiện sau 20 epoch
    )

if __name__ == "__main__":
    train_model()