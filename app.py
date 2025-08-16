from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

#load the YOLOv5
model = YOLO("yolov5s.pt")

# (COCO names)
traffic_classes = ['car', 'bus', 'truck', 'motorcycle', 'traffic light', 'stop sign']

# webcam-thingy to capture
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls]

        is_traffic = any(label in traffic_classes for label in labels)
        label_text = "Traffic Detected" if is_traffic else "No Traffic"
        color = (0, 255, 0) if is_traffic else (0, 0, 255)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # draw bounding boxes
        for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            cls_name = model.names[int(cls_id)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#send the frame 2 flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#debugging stuff
if __name__ == "__main__":
    app.run(debug=True)
