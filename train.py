import os
from ultralytics import YOLO, settings

def main():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov5n.pt")

    settings.update({"datasets_dir": os.path.join(os.path.abspath("."), "dataset")})

    # Train the model using the 'coco8.yaml' dataset for 50 epochs
    model.train(data="./data.yaml", epochs=50)
    

if __name__ == "__main__":
    main()