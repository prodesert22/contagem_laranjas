from ultralytics import YOLO

# Carrega o modelo
model = YOLO("./treinamento/yolov5/weights/best.pt")

# Faz as predições
results = model.predict(source="./test.jpg", conf=0.2)

# Pega as caixas detectadas e o nome das classes qie tem no modelo
boxes = results[0].boxes
names = results[0].names

if boxes:
    # Dicionario para contar quantos objetos de uma classe foram detectados
    count = {}
    for key in names.keys():
        count[key] = 0
    
    # Conta os objetos de cada classe e adiciona no dicionario count
    classes_objetos = boxes.cls.cpu().numpy()
    for classe in classes_objetos:
        classe = int(classe)
        count[classe] = count[classe] + 1

    for key, value in count.items():
        print(f"Total detectado: {value} {names[key]}")

# Mostra a imagem com as caixas de detecção
results[0].show()