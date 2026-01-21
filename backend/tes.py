from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model (trained on COCO or custom explosion dataset)
model = YOLO("yolov8n.pt")  # For explosion detection, use a fine-tuned model if available

# Replace with path to your video
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]
    
    # Filter explosion/fire detections by class name or ID (COCO doesn't have 'explosion', custom models do)
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = model.names[cls]

        # You can manually filter labels here (e.g., 'fire', 'explosion', etc.)
        if label.lower() in ['fire', 'explosion'] and conf > 0.5:
            print("🚨 Explosion detected!")
            # Draw the box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Explosion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
