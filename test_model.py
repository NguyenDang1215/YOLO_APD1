import traceback
from ultralytics import YOLO

yaml_path = "ultralytics/cfg/models/v8/yolov8m_goldacsim_edge.yaml"

print("🚀 Đang nổ máy động cơ YOLO-APD...")
try:
    model = YOLO(yaml_path)
    model.info()
    print("\n✅ THÀNH CÔNG RỰC RỠ!")
except Exception:
    print("\n❌ LỖI RỒI SẾP ƠI! BỘ QUÉT LỖI ĐÃ BẮT ĐƯỢC THỦ PHẠM:")
    # Hàm này sẽ in ra nguyên văn bản đồ báo lỗi từ sâu bên trong hệ thống
    traceback.print_exc()