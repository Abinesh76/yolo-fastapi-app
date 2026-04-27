from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load YOLO segmentation model
model = YOLO("yolov8n-seg.pt")

results_global = None
height = None
width = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/detect", methods=["POST"])
def detect():
    global results_global, height, width

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    height, width = image.shape[:2]

    results = model(image)
    results_global = results[0]

    objects = []

    if results_global.boxes is None or len(results_global.boxes) == 0:
        return jsonify({"objects": []})

    for i, cls in enumerate(results_global.boxes.cls):
        label = model.names[int(cls)]

        objects.append({
            "id": i,
            "name": label
        })

    return jsonify({"objects": objects})


@app.route("/mask", methods=["POST"])
def mask():
    global results_global, height, width

    if results_global is None:
        return jsonify({"error": "Upload image first"}), 400

    if "object_id" not in request.form:
        return jsonify({"error": "object_id missing"}), 400

    obj_id = int(request.form["object_id"])

    if results_global.boxes is None or obj_id >= len(results_global.boxes):
        return jsonify({"error": "Invalid object_id"}), 400

    # Use real segmentation mask if available
    if results_global.masks is not None:
        mask_data = results_global.masks.data[obj_id].cpu().numpy()
        mask_resized = cv2.resize(mask_data, (width, height))
        mask_output = (mask_resized * 255).astype(np.uint8)

    # Fallback: bounding-box mask
    else:
        box = results_global.boxes.xyxy[obj_id].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        mask_output = np.zeros((height, width), dtype=np.uint8)
        mask_output[y1:y2, x1:x2] = 255

    success, buffer = cv2.imencode(".png", mask_output)

    if not success:
        return jsonify({"error": "Failed to create mask"}), 500

    return send_file(
        BytesIO(buffer),
        mimetype="image/png"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
