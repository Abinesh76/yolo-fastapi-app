from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "YOLO API running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model(image)[0]

    detections = []

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        coords = box.xyxy[0].tolist()

        detections.append({
            "id": i,
            "label": label,
            "box": coords
        })

    return {"detections": detections}
