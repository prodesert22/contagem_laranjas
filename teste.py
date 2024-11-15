import json # biblioteca padrao do python para manipulação de arquivos json
import cv2

from deteccao_yolo import contar_objetos as contar_objetos_yolo
from deteccao_cv2 import pre_processamento, contar_objetos_opencv, contar_objetos_skimage

TOTAL_IMAGENS = 55 # total de imagens
MODELOS_YOLO = ["yolov5", "yolov8", "yolov9", "yolov11"]

def main():
    """
        Codigo para ler as imagens na pasta de teste e criar os arquivos com a contagem
        de objetos detectados e a imagens com a caixas delimitadoras
    """

    # Cria um vetor com o caminho para os arquivos das imagens que serão testadas
    imagens = []
    for i in range(1, TOTAL_IMAGENS + 1):
        imagens.append(f"./teste/imagens_originais/{i}.jpg")
    
    # Cria um vetor com as imagens pre processadas que serão usadas nas bibliotecas opencv e skimage
    imagens_pre_processadas = [] 
    for imagem in imagens:
       imagens_pre_processadas.append(pre_processamento(imagem))

    # Faz a contagem de objetos detectados e cria a imagens
    contagem_total_opencv = {}
    contagem_total_skimage = {}
    for i, imagem in enumerate(imagens_pre_processadas):
        # Usa a biblioteca opencv para fazer as detecções
        contagem_opencv, imagem_opencv = contar_objetos_opencv(imagem, imagens[i])
        # Usa a biblioteca skimage para fazer as detecções
        contagem_skimage, imagem_skimage = contar_objetos_skimage(imagem, imagens[i])
    
        contagem_total_opencv[i+1] = contagem_opencv
        contagem_total_skimage[i+1] = contagem_skimage

        # Salva a imagem nas suas respectivas pastas 
        cv2.imwrite(f"./teste/resultados/resultados_opencv/opencv_{i+1}.jpg", imagem_opencv)
        cv2.imwrite(f"./teste/resultados/resultados_skimage/skimage_{i+1}.jpg", imagem_skimage)
    
    # Salva os resultados da contagem em um arquivo json,
    # onde a chave é numero da image e o valor o o numero de objetos detectados
    with open(f"./teste/resultados/resultados_opencv/contagem_total_opencv.json", "w") as file:
        file.write(json.dumps(contagem_total_opencv, indent=2))
        
    with open(f"./teste/resultados/resultados_skimage/contagem_total_skimage.json", "w") as file:
        file.write(json.dumps(contagem_total_skimage, indent=2))
    
    for modelo in MODELOS_YOLO:
        contagem_total_yolo = {}
        for i, imagem in enumerate(imagens):
            contagem_yolo, imagem_yolo = contar_objetos_yolo(f"./treinamento/{modelo}/weights/best.pt", imagem, 0.8)
            if len(list(contagem_yolo.values())) == 0:
                contagem = 0
            else:
                contagem = list(contagem_yolo.values())[0]

            contagem_total_yolo[i+1] = contagem
            
            with open(f"./teste/resultados/resultados_yolo/{modelo}/yolo_{i+1}.jpg", "wb") as file:
                file.write(imagem_yolo)
            
        with open(f"./teste/resultados/resultados_yolo/{modelo}/contagem_total_{modelo}.json", "w") as file:
             file.write(json.dumps(contagem_total_yolo, indent=2))

if __name__ == "__main__":
    main()