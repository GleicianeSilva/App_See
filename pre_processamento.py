import os
from glob import glob
import cv2
import numpy as np


def get_imgs(path):
    return glob(os.path.join(os.getcwd(), path))  #caminho da imagem


def preprocessamento(img, tamanho=320):
    """
    Seleciona a região na imagem com a moeda e rescalona.

    Parameters
    ----------
    img : numpy.ndarray, shape=(largura, altura, 3)
        Imagem para ser preprocessada.
        
    tamanho : int, optional
        Tamanho para rescalonar a imagem de saída. The default is 320.

    Returns
    -------
    img: numpy.ndarray, shape=(tamanho, tamanho, 3)
        Imagem preprocessada e rescalonada.

    """
    
    imagem_cortada, _ = extraido_moeda(img, to_size=tamanho)
    
    if imagem_cortada is not None and len(imagem_cortada) > 0:
        return imagem_cortada[0]
    else:
        return cv2.resize(img, (tamanho, tamanho), interpolation=cv2.INTER_CUBIC) #Convertendo o tamanho da imagem para 320x320 pixels
    

def extraido_moeda(img, to_size):
    """
    Encontra as moedas na imagem e retorne uma matriz com todas as moedas no quadro (largura, altura, 3)
    
    """

    cimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# Convertendo a cor da moeda para preto e branco. img(imagem de origem)

    circles = cv2.HoughCircles(cimg, method=cv2.HOUGH_GRADIENT, dp=2, minDist=100*30, param1=250, param2=50, minRadius=30, maxRadius=400)# Encontra os círculos da moeda

    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# Convertendo a moeda para modelo de cores HSV(Matiz - faixa do arco-íris, saturação - o 'colorido' da cor e valor - brilho, ou quanta luz percebida está saindo dele)

    lower = np.array([0, 0, 0])# Definando a gama de cores ns moeda para mascara-la. # limite para canal de matiz
    upper = np.array([255, 255, 255])

    mask = cv2.blur(cv2.inRange(hsv, lower, upper), (8, 8))# Aplicando a máscara de pixels na moeda

    frames = []
    radiuses = []

    if circles is None:             # Se os círculos não foram encontrados, retorne nenhum
        return None, None

    for circle in circles[0]:

        center_x = int(circle[0])
        center_y = int(circle[1])


        if not mask[center_y, center_x]:        # Se o centro da moeda estiver na faixa de moedas mascaradas continue
            continue


        radius = circle[2] + 3    # aumenta o raio, detector de círculo tende a diminuir o raio

        radiuses.append(radius)

        x = int(center_x - radius)# Coordenadas do canto superior esquerdo do quadrado
        y = int(center_y - radius)


        if y < 0:              # À medida que o raio foi aumentando, as coordenadas poderá sair dos limites
            y = 0
        if x < 0:
            x = 0

        resized = cv2.resize(img[y: int(y + 2 * radius), x: int(x + 2 * radius)],  # Dimensiona as moedas para o mesmo tamanho
                             (to_size, to_size),
                             interpolation=cv2.INTER_CUBIC)

        frames.append(resized)

    return np.array(frames), radiuses
