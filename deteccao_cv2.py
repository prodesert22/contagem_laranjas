import cv2
import numpy as np
from cv2.typing import MatLike

caminho_imagem = "image.jpg"

def pre_processamento(caminho_imagem: str):
    # Carrega a imagem
    imagem = cv2.imread(caminho_imagem)
    cv2.imshow("imagem", imagem)
    # Converte a imagem para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)
    # Define os limites inferior e superior para a cor laranja
    limite_inferior = np.array([10, 100, 20], dtype="uint8")
    limite_superior = np.array([25, 255, 255], dtype="uint8")

    # Cria uma máscara para a cor laranja
    mascara = cv2.inRange(hsv, limite_inferior, limite_superior)
    cv2.imshow("mascara", mascara)

    # Define a cor preta e branca
    preto = (0, 0, 0)
    branco = (255, 255, 255)

    # Pinta de preto as áreas fora da máscara da laranja
    imagem[mascara == 0] = preto
    cv2.imshow("imagem mascara_invertida", imagem)
    # Pinta de branco as áreas da máscara da laranja
    #imagem[mascara == 255] = branco
    #cv2.imshow("mascara", imagem)

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

    cv2.imshow("contorno", imagem)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    filtrada = cv2.dilate(imagem, kernel, iterations = 3)
    filtrada = cv2.erode(filtrada, kernel,  iterations = 6)
    filtrada = cv2.dilate(filtrada, kernel, iterations = 4)
    filtrada = cv2.erode(filtrada, kernel,  iterations = 6)

    #cv2.imshow("filtrada", filtrada)
    #cv2.waitKey(0)

    filtrada = cv2.cvtColor(filtrada, cv2.COLOR_BGR2GRAY)
    cv2.imshow("filtrada", filtrada)
    return filtrada

def contar_objetos_opencv(imagem: MatLike, caminho_imagem: str):
    _num_obj, _, stats, centroids = cv2.connectedComponentsWithStats(imagem, connectivity=4, ltype=cv2.CV_32S)
    
    img_com_caixa = cv2.imread(caminho_imagem)
    
    num_obj = 0
    for i in range(_num_obj):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        
        height, width, _ = img_com_caixa.shape

        if area >= 100 and w * h < height * width:
            cv2.rectangle(img_com_caixa, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(img_com_caixa, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            num_obj += 1

    return num_obj, img_com_caixa
    
def contar_objetos_skimage(imagem: MatLike, caminho_imagem: str):
    from skimage.measure import label, regionprops

    # Encontrar componentes conectados
    label_image = label(imagem)

    # Criar uma cópia da imagem original para desenhar os retângulos
    img_com_caixa = cv2.imread(caminho_imagem)

    # Iterar sobre as regiões encontradas
    num_obj = 0
    for region in regionprops(label_image):
        if region.area >= 100:
            # Obter as coordenadas da região
            minr, minc, maxr, maxc = region.bbox

            # Desenhar o retângulo na imagem
            cv2.rectangle(img_com_caixa, (minc, minr), (maxc, maxr), (0, 255, 0), 2)  # Verde, 2 pixels de espessura

            num_obj += 1
    
    return num_obj, img_com_caixa

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