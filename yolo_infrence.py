from ultralytics import YOLO
model = YOLO(r"models\yolov8x.pt")  # load a pretrained model (recommended for training)
model.predict(source="input_videos/input_video.mp4", show=True, save=True)  
