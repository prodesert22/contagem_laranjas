import tempfile
import os
from ultralytics import YOLO

def contar_objetos(modelo: str, imagem: str, conf=0.5):
    # Carrega o modelo
    model = YOLO(modelo)

    # Faz as predições
    results = model.predict(source=imagem, conf=conf)

    # Pega as caixas detectadas e o nome das classes que tem no modelo
    boxes = results[0].boxes # objeto com as caixas delimitadoras
    names = results[0].names # dicionarios com id e nome de cada classe

    count: dict[int, int] = {}
    if boxes:
        # Dicionario para contar quantos objetos de uma classe foram detectados
        # Enche o dicionarios com 0 para cada classe
        for key in names.keys():
            count[key] = 0

        # Conta os objetos de cada classe e adiciona no dicionario count
        classes_objetos = boxes.cls.cpu().numpy()
        for classe in classes_objetos:
            classe = int(classe)
            count[classe] = count[classe] + 1

    # Salva a imagem com as caixas de detecção
    fd, temp_filename = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    
    # Salva a image em um arquivo temporario
    results[0].save(temp_filename)
    # Le o arquivo temporario da imagem e retorna ele em binario
    with open(temp_filename, "rb") as file:
        imagem_final = file.read()

    contagem_final: dict[str, int] = {}
    for key, value in count.items():
        name = names[key]
        contagem_final[name] = value

    return contagem_final, imagem_final

if __name__ == "__main__":
    contar_objetos("./treinamento/yolov5/weights/best.pt", "laranjas.jpg")
