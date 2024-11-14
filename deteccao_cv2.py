import cv2
import numpy as np
from cv2.typing import MatLike

caminho_imagem = "laranja.jpg"

def pre_processamento(caminho_imagem: str):
    # Carrega a imagem
    imagem = cv2.imread(caminho_imagem)
    #cv2.imshow("imagem", imagem)
    # Converte a imagem para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV", hsv)
    # Define os limites inferior e superior para a cor laranja
    limite_inferior = np.array([10, 100, 20], dtype="uint8")
    limite_superior = np.array([25, 255, 255], dtype="uint8")

    # Cria uma máscara para a cor laranja
    mascara = cv2.inRange(hsv, limite_inferior, limite_superior)
    #cv2.imshow("mascara", mascara)

    # Define a cor preta e branca
    preto = (0, 0, 0)
    branco = (255, 255, 255)

    # Pinta de preto as áreas fora da máscara da laranja
    imagem[mascara == 0] = preto
    #cv2.imshow("imagem mascara_invertida", imagem)

    # Encontra os contornos da máscara da laranja
    # É criado os contornos para preencher as falhas no inteiror da laranja
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Itera sobre os contornos
    for contorno in contornos:
        # Calcula a área do contorno
        area = cv2.contourArea(contorno)

        # Define um limite de área para evitar detectar pequenos ruídos
        if area > 100:
            # Cria uma máscara para o contorno atual
            mascara_contorno = np.zeros(imagem.shape[:2], dtype="uint8")
            cv2.drawContours(mascara_contorno, [contorno], -1, 255, -1)

            # Pinta de branco os pixels dentro do contorno
            imagem[mascara_contorno == 255] = branco

    #cv2.imshow("contorno", imagem)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    img_processada = cv2.dilate(imagem, kernel, iterations = 3)
    img_processada = cv2.erode(img_processada, kernel,  iterations = 6)
    img_processada = cv2.dilate(img_processada, kernel, iterations = 4)
    img_processada = cv2.erode(img_processada, kernel,  iterations = 6)

    img_processada = cv2.cvtColor(img_processada, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("img processada", img_processada)
    return img_processada

def contar_objetos_opencv(imagem: MatLike, caminho_imagem: str):
    # Encontrar componentes conectados
    _num_obj, _, stats, _ = cv2.connectedComponentsWithStats(imagem, connectivity=4, ltype=cv2.CV_32S)
    
    # Abre a imagem original para criar as caixas de detecção
    img_com_caixa = cv2.imread(caminho_imagem)
    
    # Como é filtrado alguns objetos é necessario criar uma variavel nova para contar
    # o numero de objetos detectados
    num_obj = 0
    for i in range(_num_obj):
        # Pega o ponto x e y do canto superior esquerdo do objeto
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        # Pega a largunta e altura do objeto
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # Pega area
        area = stats[i, cv2.CC_STAT_AREA]
        
        height, width, _ = img_com_caixa.shape

        # Algumas imagens tem objetos detectados como a imagem inteira então o if abaixo ignora eles
        # Filtra objetos com a area maior que 100 pixeis
        if area >= 100 and w * h < height * width:
            # Desenha a caixa de detecção
            cv2.rectangle(img_com_caixa, (x, y), (x + w, y + h), (0, 255, 0), 2)
            num_obj += 1

    return num_obj, img_com_caixa
    
def contar_objetos_skimage(imagem: MatLike, caminho_imagem: str):
    from skimage.measure import label, regionprops

    # Encontrar componentes conectados
    label_image = label(imagem)

    # Abre a imagem original para criar as caixas de detecção
    img_com_caixa = cv2.imread(caminho_imagem)

    num_obj = 0
    for region in regionprops(label_image):
        # Filtra objetos com a area maior que 100 pixeis
        if region.area >= 100:
            # Obtem as coordenadas da região
            minr, minc, maxr, maxc = region.bbox

            # Desenha a caixa de detecção
            cv2.rectangle(img_com_caixa, (minc, minr), (maxc, maxr), (0, 255, 0), 2)  # Verde, 2 pixels de espessura

            num_obj += 1
    
    return num_obj, img_com_caixa

if __name__ == "__main__":
    img_processada = pre_processamento(caminho_imagem)

    num_obj_cv2, img_cv2 = contar_objetos_opencv(img_processada, caminho_imagem)

    num_obj_skimage, img_skimage = contar_objetos_skimage(img_processada, caminho_imagem)

    # Exibir a imagem com os retângulos
    print(f"cv2 - total de lanrajas: {num_obj_cv2}")
    cv2.imshow("Imagem com Retangulos cv2", img_cv2)
    print(f"scikit-image - total de lanrajas: {num_obj_skimage}")
    cv2.imshow("Imagem com Retangulos scikit-image", img_skimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()