from ultralytics import YOLO

model = YOLO("./models/YOLOv8n_100_Epoch_Dataset1/best.pt")
results = model.predict(source="./tests/image.png", conf=0.01, iou=0.4, save=True)

class_names = ['can', 'glass', 'plastic']

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x_center, y_center, width, height = box.xywh[0]
        label = class_names[cls_id]
        print(f"Label: {label}, Confidence: {conf:.3f}, Box Center: ({x_center:.1f}, {y_center:.1f}), Size: ({width:.1f}, {height:.1f})")
