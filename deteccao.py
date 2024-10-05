from ultralytics import YOLO

model = YOLO("./treinamento/yolov5/weights/best.pt")

results = model.predict(source="./image.jpg", conf=0.5)

boxes = results[0].boxes
names = results[0].names

if boxes:
    count = {}
    for key in names.keys():
        count[key] = 0
    
    classes_objetos = boxes.cls.cpu().numpy()
    for classe in classes_objetos:
        classe = int(classe)
        count[classe] = count[classe] + 1

    for key, value in count.items():
        print(f"Total detectado: {value} {names[key]}")

results[0].show()