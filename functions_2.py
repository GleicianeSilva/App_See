from pathlib import Path
import skimage.io as io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import transform
from tqdm.notebook import tqdm
import pathlib
import cv2
from sklearn.model_selection import train_test_split

#Acessando a pasta data
DataPath = pathlib.Path("data")

#Pegando as imagens de todas as pastas
image_files= DataPath.glob("*/*.png")

#definindo o tamahno das imagens
im_size = 320


def read_file(fname):
    # Lendo a imagem
    im = Image.open(fname)

    #redimicionando o tamanho da imagem. Transformando a imagem de 320 por 320
    im.thumbnail((im_size, im_size))

    # Convertendo em array de número
    im_array = np.asarray(im)

    # Divide o nome do arquivo e pega o valor da moeda e transforma em uma string
    target = str(fname.stem.split('_')[0])

    #Retorna a imagem o valor da moeda
    return im_array, target

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
def read_file_2(fname):

    imgs = np.empty((0, 128, 128, 3), dtype=np.float32)

    for img_path in fname:
        img = io.imread(img_path)
        img = transform.resize(img, (128, 128))
        imgs = np.insert(imgs, 0, img, 0)

    X_train = imgs

    image = X_train[467]

    return image
def quadradinho(contourn):
    # São as listas que receberão as imagens e os valores da moedas
    images = []
    targets = []
    image, target = read_file_2(image_files)

    #Percorrer todas as imagens que estão nas pastas
    for image_file in tqdm(image_files):

        #Chamar a função read_file

        float(target.replace(',', '.'))

        #Adiciona a imagem e os valora das moedas a lista
        images.append(image)
        targets.append(target)

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    X = np.array((images), dtype=np.object)
    #Tamanho de convulução
    #X = (np.array(images).astype(np.int) / 127.5) - 1 #probleminha

    #Transforma a lista de valores de moedas em um array numpy
    y_cls = np.array(targets)

    print(X.shape, y_cls.shape)

    i = 0
    #plt.imshow(np.uint8((X[i] + 1) * 127.5))

    cv2.imshow('img',contourn)
    plt.title(str(y_cls[i]));