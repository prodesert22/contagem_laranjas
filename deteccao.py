from ultralytics import YOLO

def contar_objetos(modelo: str, imagem: str, conf=0.5):
    # Carrega o modelo
    model = YOLO(modelo)

    # Faz as predições
    results = model.predict(source=imagem, conf=conf)

    # Pega as caixas detectadas e o nome das classes qie tem no modelo
    boxes = results[0].boxes
    names = results[0].names

    count = {}
    if boxes:
        # Dicionario para contar quantos objetos de uma classe foram detectados
        for key in names.keys():
            count[key] = 0
        # Conta os objetos de cada classe e adiciona no dicionario count
        classes_objetos = boxes.cls.cpu().numpy()
        for classe in classes_objetos:
            classe = int(classe)
            count[classe] = count[classe] + 1

    # Mostra a imagem com as caixas de detecção
    results[0].save("resultado_" + imagem)
    return count, "resultado_" + imagem

if __name__ == "__main__":
    contar_objetos("./treinamento/yolov5/weights/best.pt", "laranjas.jpg")
