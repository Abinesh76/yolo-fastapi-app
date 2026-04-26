const upload = document.getElementById("upload");
const dropdown = document.getElementById("dropdown");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let detections = [];
let image = new Image();

upload.addEventListener("change", async () => {
  const file = upload.files[0];

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/detect", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  detections = data.detections;

  image.src = URL.createObjectURL(file);
  image.onload = () => {
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);
  };

  dropdown.innerHTML = "<option>Select Object</option>";

  detections.forEach(det => {
    const option = document.createElement("option");
    option.value = det.id;
    option.text = det.label + " (" + det.id + ")";
    dropdown.appendChild(option);
  });
});

dropdown.addEventListener("change", () => {
  const selectedId = parseInt(dropdown.value);

  ctx.drawImage(image, 0, 0);

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.box;

    if (det.id !== selectedId) {
      ctx.fillStyle = "black";
      ctx.globalAlpha = 0.7;
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    } else {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  });

  ctx.globalAlpha = 1;
});
