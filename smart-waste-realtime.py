from ultralytics import YOLO
import cv2

model = YOLO("./models/YOLOv8s_100_Epoch_DatasetMerge_NewAugmentation/best.pt")
class_names = ['can', 'glass', 'plastic']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, iou=0.6, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x_center, y_center, width, height = box.xywh[0]
            label = class_names[cls_id]
            print(f"Label: {label}, Confidence: {conf:.3f}, Box Center: ({x_center:.1f}, {y_center:.1f}), Size: ({width:.1f}, {height:.1f})")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
