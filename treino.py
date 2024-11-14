import os
from ultralytics import YOLO, settings

def main():
    # Carrega o modelo pretreinado do YOLO
    # YOLO 5 - yolov5nu.pt
    # YOLO 8 - yolov8n.pt
    # YOLO 9 - yolov9c.pt
    # YOLO 11 - yolo11n.pt
    model = YOLO("yolov9c.pt")

    settings.update({"datasets_dir": os.path.join(os.path.abspath("."), "dataset")})

    # Treina o modelo usando o dataset de laranjas
    model.train(data="./data.yaml", epochs=100)


if __name__ == "__main__":
    main()