from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

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
    filepath = "input.jpg"
    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    height, width = image.shape[:2]

    results = model(image)
    results_global = results[0]

    objects = []

    if results_global.boxes is None:
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
        return jsonify({"error": "Please upload image first"}), 400

    obj_id = int(request.form["object_id"])

    if results_global.masks is not None:
        mask_data = results_global.masks.data[obj_id].cpu().numpy()

        mask_resized = cv2.resize(mask_data, (width, height))
        mask_output = (mask_resized * 255).astype(np.uint8)

    else:
        box = results_global.boxes.xyxy[obj_id].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        mask_output = np.zeros((height, width), dtype=np.uint8)
        mask_output[y1:y2, x1:x2] = 255

    output_path = "mask.png"
    cv2.imwrite(output_path, mask_output)

    return send_file(output_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
