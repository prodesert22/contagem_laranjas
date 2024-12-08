import io
from PIL import Image
from ultralytics import YOLO

def contar_objetos(modelo: str, imagem: str, conf=0.5):
    # Carrega o modelo
    model = YOLO(modelo)

    # Faz as predições
    results = model.predict(source=imagem, conf=conf, show_labels=False, show_boxes=True)

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

    imagem_bgr = results[0].plot(labels=False)
    im_rgb = Image.fromarray(imagem_bgr[..., ::-1])

    # Salva a imagem como bytes
    buffer = io.BytesIO()
    im_rgb.save(buffer, format="JPEG")
    imagem_final = buffer.getvalue()
    buffer.close()

    contagem_final: dict[str, int] = {}
    for key, value in count.items():
        name = names[key]
        contagem_final[name] = value

    return contagem_final, imagem_final

if __name__ == "__main__":
    # Seleciona modelo usado e a imagem para ser detectada
    contagem, imagem = contar_objetos("./treinamento/yolov5/weights/best.pt", "docs/laranja.png")
    
    print(f"Contagem de laranjas: {contagem}")
    with open("imagem_teste.jpg", "wb") as file:
        file.write(imagem)
