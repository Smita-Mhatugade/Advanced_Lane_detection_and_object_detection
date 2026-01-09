!pip install roboflow
!pip install ultralytics
from roboflow import Roboflow
from ultralytics import YOLO
rf = Roboflow(api_key="erIG2xtnJhFIcGEISkan")
project = rf.workspace("wildlife-project-rnl6y").project("poaching-ready-brfz6")
version = project.version(2)
dataset = version.download("yolov8")
model = YOLO("yolov8n.pt")  # lightweight base model
model.train(data=dataset.location + "/data.yaml", epochs=50, imgsz=640)

print("✅ Training complete! Check runs/detect/train/weights/best.pt")