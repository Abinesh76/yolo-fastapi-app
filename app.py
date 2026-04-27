import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

def detect_and_mask(image, selected_object):

    results = model(image)[0]

    objects = []
    boxes = []

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        objects.append(f"{i}-{label}")
        boxes.append(box.xyxy[0].cpu().numpy())

    if selected_object is None:
        return image, objects

    obj_id = int(selected_object.split("-")[0])
    box = boxes[obj_id]

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = map(int, box)
    mask[y1:y2, x1:x2] = 255

    result_img = image.copy()
    result_img[mask == 0] = 0

    return result_img, objects


with gr.Blocks() as demo:

    gr.Markdown("## YOLO Object Detection + Mask")

    image_input = gr.Image(type="numpy")
    dropdown = gr.Dropdown(label="Select Object", choices=[])
    output_image = gr.Image()

    btn = gr.Button("Detect")

    def detect(image):
        results = model(image)[0]

        objects = []
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            objects.append(f"{i}-{label}")

        return objects

    btn.click(detect, inputs=image_input, outputs=dropdown)

    dropdown.change(detect_and_mask,
                    inputs=[image_input, dropdown],
                    outputs=[output_image, dropdown])

demo.launch()
