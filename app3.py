from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

# Try to import the Raspberry Pi camera module
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False

app = Flask(__name__)

# Load the YOLOv5 model (force CPU because Raspberry Pi has no CUDA)
model = YOLO("yolov5s.pt")
model.fuse()  # optional: speeds up inference on CPU

traffic_classes = ['car', 'bus', 'truck', 'motorcycle', 'traffic light', 'stop sign']

# Initialize camera
if USE_PICAMERA:
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam.configure(config)
    picam.start()
else:
    cap = cv2.VideoCapture(0)  # USB webcam fallback
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def gen_frames():
    while True:
        # get a frame from the camera
        if USE_PICAMERA:
            frame = picam.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        # Inference
        results = model(frame, verbose=False)
        for r in results:
            annotated = r.plot()  # draw boxes and labels on a copy of the image

        # encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # use host=0.0.0.0 so you can connect from other devices
    app.run(host="0.0.0.0", port=5000, debug=False)
