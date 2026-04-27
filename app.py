from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n-seg.pt")

results_global = None
height = None
width = None


# HOME PAGE
@app.route('/')
def home():
    return render_template("index.html")


# HEALTH CHECK
@app.route('/health')
def health():
    return jsonify({"status": "ok"})


# DETECT OBJECTS
@app.route('/detect', methods=['POST'])
def detect():
    global results_global, height, width

    file = request.files['file']

    filepath = "input.jpg"
    file.save(filepath)

    image = cv2.imread(filepath)
    height, width = image.shape[:2]

    results = model(image)
    results_global = results[0]

    objects = []

    for i, cls in enumerate(results_global.boxes.cls):
        label = model.names[int(cls)]

        objects.append({
            "id": i,
            "name": label
        })

    return jsonify({"objects": objects})


# MASK OBJECT
@app.route('/mask', methods=['POST'])
def mask():
    global results_global, height, width

    obj_id = int(request.form['object_id'])

    box = results_global.boxes.xyxy[obj_id].cpu().numpy()

    mask = np.zeros((height, width), dtype=np.uint8)

    x1, y1, x2, y2 = map(int, box)

    mask[y1:y2, x1:x2] = 255

    output_path = "mask.png"
    cv2.imwrite(output_path, mask)

    return send_file(output_path, mimetype='image/png')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
