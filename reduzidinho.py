import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from random import shuffle
import time

DataPath = pathlib.Path("datasets") # acessa a pasta raiz
todos_os_caminhos = DataPath.glob("*.jpg") # pega todas as imagens de todas as pastas
todos_os_caminhos = list(map(lambda x : str(x) , todos_os_caminhos)) #converte todos os objetos da lista para string

shuffle(todos_os_caminhos) #randomizamos as imagens

''' decodificamos as imagens para jpeg e ficamos só com as imagens que não dão erro, descartamos o resto '''

def TestaAQualidadeDaImagem(todos_os_caminhos):
    caminhos_novos = [] #declaramos uma variavel que vai receber todos os caminhos das imagens que funcionam
    for caminho in tqdm(todos_os_caminhos): #para cada caminho em todos os caminhos
        try : #eu tento fazer 2 coisas
            image = tf.io.read_file(caminho) #primeira: ler a imagem
            image = tf.io.decode_jpeg(image , channels = 3) #segunda: decodificar a imagem
        except : #se eu não conseguir
            continue #eu deixo pra lá, e continuo
        caminhos_novos.append(caminho) # se eu conseguir, eu adiciono o caminho da imagem à variável caminhos_novos
    return caminhos_novos #retorno a variável caminhos_novos.

todos_os_caminhos = TestaAQualidadeDaImagem(todos_os_caminhos) #chama a função e joga o resultado em todos_os_caminhos
#print("\n%s imagens foram removidas da variavel todos_os_caminhos"%(qtinicial-len(todos_os_caminhos)))

# pega os labels das variáveis contidas em todos_os_caminhos
def classificar_rotulo(image_path):
    return image_path.split("_")[-2] #divido a string do caminho da imagem na '/' e pego o penúltimo index

todos_os_rotulos = list(map(lambda x : classificar_rotulo(x) , todos_os_caminhos)) #faz um for e chamo a função e retorno uma lista

#depois, usamos o label encoder do sklearn para atribuir rotulos numericos a cada categoria de moeda
from sklearn.preprocessing import LabelEncoder 
Le = LabelEncoder()
todos_os_rotulos = Le.fit_transform(todos_os_rotulos) #transforma a categoria em um número
print(todos_os_rotulos)

#depois, usamos outra biblioteca do sklearn para separar nossa base de dados em teste e validação
from sklearn.model_selection import train_test_split 

caminhos_treino , caminhos_validacao , rotulos_treino , rotulos_validacao = train_test_split(todos_os_caminhos , todos_os_rotulos) #separa o dataset em validação (teste) e treino

''' carrega a imagem, lê, decodifica, e retorna a imagem e o rótulo '''
def carregaImagem(image , rotulo):
    image = tf.io.read_file(image) #lê a imagem e salva em imagem
    image = tf.io.decode_jpeg(image , channels = 3) #decodifica a imagem e salva em imagem
    return image , rotulo #retorna imagem e rotulo

TAMANHO_DA_IMAGEM = 320  #definição do tamanho da imagem !importante!
BATCH_SIZE = 3059 #define por causa da performance, para que fique mais fácil e rápido processar todas as imagens

# transformamos a imagem pelo tamanho da imagem, transformando ela em quadrada também
resize = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(TAMANHO_DA_IMAGEM, TAMANHO_DA_IMAGEM)          
])

# aumento do nosso dataset -> adicionamos imagens rotacionadas na horizontal, vertical, por angulo, com aumento e diminuição do zoom
aumento_do_dataset_de_treino = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = (-0.3, -0.2))
])

# função usada para criar um objeto de dados do tensorflow (converter nosso dataset para algo que o tensorflow consegue entender)
filtro_automatico = tf.data.experimental.AUTOTUNE
def pega_dataset(caminhos , rotulos , train = True):
    caminho_da_imagem = tf.convert_to_tensor(caminhos) #conversao caminhos imagens
    rotulo_da_imagem = tf.convert_to_tensor(rotulos) #conversao rotulos imagens

    image_dataset = tf.data.Dataset.from_tensor_slices(caminho_da_imagem) #conversao imagens do dataset inteiro
    rotulos_dataset = tf.data.Dataset.from_tensor_slices(rotulo_da_imagem) #conversao rotulos do dataset inteiro

    dataset = tf.data.Dataset.zip((image_dataset , rotulos_dataset)) #associa as imagens com os rotulos e define que isso é o dataset único

    dataset = dataset.map(lambda image , label : carregaImagem(image , label)) #mapeia usando a função carregaimagem
    dataset = dataset.map(lambda image, label: (resize(image), label) , num_parallel_calls=filtro_automatico) #usamos o filtro para processar a imagem
    dataset = dataset.shuffle(1000) #aleatorizamos novamente nosso dataset
    dataset = dataset.batch(BATCH_SIZE) #o TAMANHO_DO_LOTE tem que ser um número divisível pelo tamanho do dataset

    if train: #se for o dataset de treino
        dataset = dataset.map(lambda image, label: (aumento_do_dataset_de_treino(image), label) , num_parallel_calls=filtro_automatico) #aumento o tamanho da minha base de testes
        dataset = dataset.repeat() #repito o aumento do dataset
    return dataset #retorno o dataset


# crio o objeto de dataset de treino e verifico se ele está bom

dataset_de_treino = pega_dataset(caminhos_treino , rotulos_treino) #criando o dataset de treino

image , label = next(iter(dataset_de_treino)) #pegando uma imagem e um label qualquer (o próximo da iteração nesse caso)
print(image.shape) #pegamos o objeto de lista de 4 dimensões com 94 imagens de 224x224 com 3 cores
print(label.shape) #pegamos o objeto de lista com todos os rótulos

# só coloquei para a gente enxergar
print(Le.inverse_transform(label)[0]) #escrevendo o valor da moeda
plt.imshow((image[0].numpy()/255).reshape(320 , 320 , 3)) #plotando a imagem


dataset_de_validacao = pega_dataset(caminhos_validacao , rotulos_validacao , train = False) #mostro o tempo de execução

image , label = next(iter(dataset_de_validacao)) #pego uma moeda do dataset de validação
print(image.shape) #imprimimos o tamanho do dataset de imagens (32 imagens, 224x224 tamanho, 3 cores RGB vermelho, verde, azul)
print(label.shape) #imprimimos a quantidade de rótulos

# plotamos a primeira imagem para enxergar nosso dataset de validação
print(Le.inverse_transform(label)[0]) #printo o rótulo da moeda
plt.imshow((image[0].numpy()/255).reshape(320 , 320 , 3)) #ploto a imagem da moeda

"""#processamento de imagem (única parte que não comentamos, mas que toda ela tem a explicação na documentação do tensorflow keras https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br"""

# Model part 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization , GlobalMaxPool2D

model = Sequential() 

# Block 1 
model.add(Conv2D(input_shape=(320 , 320 , 3),  padding='same',filters=32, kernel_size=(7, 7))) #kernel = máscara de 7x7; filters = 32
model.add(Activation('relu')) #pesquisar
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(filters=64,  padding='valid', kernel_size=(5, 5)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block 3 
model.add(Conv2D(filters=128, padding='valid', kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Block 4 
model.add(Conv2D(filters=256, padding='valid', kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256 , kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(GlobalMaxPool2D())

model.add(Dense(units=256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))

print(model.summary())

# retirado da documentação do tensorflow https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# aqui ele começa a treinar o nosso modelo
history = model.fit(
    dataset_de_treino,
    steps_per_epoch=len(caminhos_treino),
    epochs=2000,
    validation_data=dataset_de_validacao,
    validation_steps = len(caminhos_validacao), batch_size=32,
)

# Evaluating the loaded model
perda, acuracia = model.evaluate(dataset_de_validacao) #passei meu dataset de validacao para o meu modelo avaliar

print(" testando a acuracia : " , acuracia)
print(" testando a perda : " , perda)

dataset_de_validacao #para ver a variável do dataset de validação