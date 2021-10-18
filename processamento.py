import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from random import shuffle
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Imports da bibliotecas de rede neural
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Activation, Conv2D, BatchNormalization , MaxPooling2D, GlobalMaxPooling2D
from  tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


from pre_processamento import preprocessamento



'''----------------------------------------------------------------------'''
'''' Definições de Variáveis  '''

todos_os_caminhos_original = glob.glob("datasets/*.jpg") # Carregando toda as imagens da pasta raiz e pega todas as imagens dentro da pasta datasets
todos_os_caminhos_original = list(map(lambda x : str(x) , todos_os_caminhos_original)) # Converte todos os objetos da lista para string

shuffle(todos_os_caminhos_original) # Randomiza as imagens

TAMANHO_DA_IMAGEM = 180 #definição do tamanho da imagem.


#Tamanho do lote define o número de amostras que serão propagadas pela rede. batch_size: Inteiro ou Nenhum. Número de amostras por atualização de gradiente.

BATCH_SIZE = 16    #define por causa da performance, para que fique mais fácil e rápido processar todas as imagens. 3059 é a quantidade de imagem que se tem na pasta Datasets e dentro dela existe 5 tipo de moedas
NUM_EPOCHS_1 = 20 # Quantidade de épocas para a primeira parte do treinameto
NUM_EPOCHS_2 = 50 # Quantidade de épocas para a segunda parte do treinameto
learning_rate_1 = 0.001 # Taxa de aprendizagem para a primeira parte do treinameto
learning_rate_2 = 0.000001 # Taxa de aprendizagem para a segunda parte do treinameto
imagem_teste = '25_20.jpg'



'''----------------------------------------------------------------------'''
'''' Definiçção de funções auxiliares  '''

# Pega os labels das variáveis contidas em todos_os_caminhos
def classificar_rotulo(image_path):
    """
        Transforma o caminho da imagem no rótulo da classe/moeda.
    """
    return int(image_path.split("_")[0].split(os.sep)[-1]) #Dividido a string do caminho da imagem na '_' e pega o penúltimo index

def carregaImagem(image, tensor=True):
    """
        Carrega a imagem, aplica o preprocessamento com a opencv, redimeniona e retorna uma imagem em formato
        de matriz ou Tensor
    """

    img = cv2.imread(image, cv2.IMREAD_COLOR) # Carrega a imagem em formato de matriz.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #  Converte a imagem BGR em uma imagem RGB
    img = preprocessamento(img, tamanho=TAMANHO_DA_IMAGEM)# Recorta a imagem paa a região da moeda e a reescala.
    
    if tensor:
        return tf.constant(img, dtype='uint8') # Converte a representação matricial para a Tensor.'''

    return img # retorna imagem representada em Tensor


def pega_dataset(caminhos, rotulos , train = True):
    """
        Cria um objeto dataset do pacote tensorflow com as imagens especificadas em `caminhos`. Cada imagem é carregada
        utilizando a função `carregaImagem`. Também é aplicada dataugmentation.
    """


# aumento do dataset -> adicionamos imagens rotacionadas na horizontal, vertical, por angulo, com aumento e diminuição do zoom
    aumento_do_dataset_de_treino = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = (-0.3, -0.2))
    ])

    # função usada para criar um objeto de dados do tensorflow (converter o dataset para algo que o tensorflow consegue entender)
    filtro_automatico = tf.data.experimental.AUTOTUNE

    caminho_da_imagem = caminhos #tf.convert_to_tensor(caminhos) #conversao caminhos imagens
    rotulo_da_imagem = tf.convert_to_tensor(rotulos) #conversao rotulos imagens

    image_dataset = tf.data.Dataset.from_tensor_slices(list(map(carregaImagem, caminho_da_imagem))) #conversao imagens do dataset inteiro
    rotulos_dataset = tf.data.Dataset.from_tensor_slices(rotulo_da_imagem) #conversao rotulos do dataset inteiro

    image_dataset = image_dataset.map(lambda x: x / 255) #Normalização do dataset

    dataset = tf.data.Dataset.zip((image_dataset, rotulos_dataset)) #associa as imagens com os rotulos e define que isso é o dataset único

    dataset = dataset.batch(BATCH_SIZE) #o TAMANHO_DO_LOTE tem que ser um número divisível pelo tamanho do dataset

    if train: #se for o dataset de treino
        dataset = dataset.repeat() #Repetido o aumento do dataset
    return dataset #retorno o dataset


'''----------------------------------------------------------------------'''
'''' Início  '''

todos_os_caminhos = todos_os_caminhos_original
todos_os_rotulos = list(map(lambda x : classificar_rotulo(x) , todos_os_caminhos)) #faz um for e chamo a função e retorno uma lista

#depois, usa label encoder do sklearn para atribuir rotulos numericos a cada categoria de moeda

Le = LabelEncoder()

todos_os_rotulos = Le.fit_transform(todos_os_rotulos) #transforma a categoria em um número

print('\n', todos_os_rotulos)

print("\nQuantidade de Moeda: ", len(todos_os_rotulos) )

print("\nQuantidade de Classe: ", len(np.unique(todos_os_rotulos)))

[print(f"\nMoeda de {Le.inverse_transform((cl, ))[0]}: {qtd}") for cl, qtd in zip(*np.unique(todos_os_rotulos, return_counts=True))]


#depois, usa outra biblioteca do sklearn para separar a base de dados em teste e validação

caminhos_treino, caminhos_validacao, \
    rotulos_treino, rotulos_validacao = train_test_split(todos_os_caminhos, todos_os_rotulos,
                                                         test_size=0.3) #separa o dataset em validação (teste) e treino

print("\nQuantidade de Moeda Treinada: ", len(caminhos_treino))
print("\nQuantidade de Moedas Validada: ", len(caminhos_validacao))



# Criando o objeto de dataset de treino e verificando se ele está bom

dataset_de_treino = pega_dataset(caminhos_treino, rotulos_treino) #criando o dataset de treino

image, label = next(iter(dataset_de_treino)) #Pegando uma imagem e um label qualquer (o próximo da iteração nesse caso)
print("\nTamanho do Datasets de Treino: ", image.shape) #pegando o objeto de lista de 4 dimensões com 94 imagens de 320x320 com 3 cores
print("\nQuantidade de Rotulos de Treino: ", label.shape) #pegando o objeto de lista com todos os rótulos

coin_values = Le.inverse_transform(label)#Escrevemdo o valor da moeda
print('\nPrimeira Moeda do Dataset de Treino: ', coin_values[0])

# Mostra primeira imagem do dataset de treinamento
plt.figure()
plt.imshow(image.numpy()[0])
plt.title(f'Primeira imagem treinada - Moeda de: {Le.inverse_transform([label[0].numpy()])[0]}')


dataset_de_validacao = pega_dataset(caminhos_validacao , rotulos_validacao , train = False) #mostro o tempo de execução

image, label = next(iter(dataset_de_validacao)) #pegando uma moeda do dataset de validação
print('\nTamanho do Datasets de Validação: ', image.shape) #imprimido tamanho do dataset de imagens (32 imagens, 320x320 tamanho, 3 cores RGB vermelho, verde, azul)
print('\nQuantidade de Rotulos de Validação: ', label.shape) #imprimido a quantidade de rótulos


#Plotando a primeira imagem para enxergar o dataset de validação
coin_values = Le.inverse_transform(label)#Escrevendo o valor da moeda
print('\nPrimeira Moeda do Dataset de Validação: ', coin_values[0])#printa o rótulo da moeda

# Mostra primeira imagem do dataset de validação
plt.figure()
plt.imshow(image.numpy()[0])#ploto a imagem da moeda
plt.title(f'Primeira imagem validada - Moeda de: {Le.inverse_transform([label[0].numpy()])[0]}')


"""Processamento de Imagem (Explicação na documentação do tensorflow keras https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br"""

#Parte do modelo

'''----------------------------------------------------------------------'''


'''----------------------< Arquitetura da Rede Neural >----------------------'''


''' ----------------------< Contruindo o Modelo da Rede Neural >----------------------'''

# %%

print('\n')

model = Sequential(name='ModeloMoedas') #Estrutura da Rede Neural

#Rede CNN
#Camada adicionada ao shape de entrada
# Pode ser feito alteração de inicializacao, bias, entre outros -- https://keras.io/layers/convolutional/#conv2d

#'Relu' --> Função de ativação, utilizada para treinar mais rápido.

model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                 input_shape=(TAMANHO_DA_IMAGEM, TAMANHO_DA_IMAGEM, 3))) #Rede Convolucional recebendo a primeira imagem, mas a camada de ativação 'Relu'
model.add(MaxPooling2D((3, 3))) #Camada Pooling

model.add(Conv2D(8, (3, 3), activation='relu', padding='same')) #Rede Convolucional recebendo dados da outra rede, mas a camada de ativação 'Relu'
model.add(MaxPooling2D(pool_size=(3, 3))) #Camada Pooling

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(GlobalMaxPooling2D())
model.add(BatchNormalization())

#Rede MLP
#Camada totalmente conectada
model.add(Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.001))) #Camada densa
model.add(Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.001))) #Camada densa
model.add(Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.001))) #Camada densa
model.add(Dense(5, activation='softmax') )#Camada de saida com o resultado das classes. Camada densa, mas a camada de ativação para classificar


print(model.summary())  # Print do Sumário do Modelo


#retirado da documentação do tensorflow https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br

'''----------------------< Compilação do Modelo >----------------------'''
# Compilando a rede definindo: otimizador, metrica e loss function(função de perda)
# adam --> otimizador
# sparse_categorical_crossentropy --> função de perda para classificação de multiclasse
# accuracy --> métrica de acerto
model.compile(optimizer= Adam(learning_rate=learning_rate_1), loss='sparse_categorical_crossentropy',metrics=['accuracy'])


''' ----------------------< Treinamento do Modelo >----------------------'''
reduce_LRO = ReduceLROnPlateau(patience=5, factor=0.01, verbose=True)
checkpoint = ModelCheckpoint('chest_orientation_model.hdf5', monitor='val_loss', verbose=1, mode='min', save_best_only=True) #ModelCheckPoint para salvar o modelo que tiver o melhor loss durante o treinamento
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=12, mode='min', verbose=1) #EarlyStop para interromper o treinamento caso a rede pare de aprender.

#Começando a primeira parte do treinamento do modelo
#epochs:o número de vezes que o modelo é treinado em todo o conjunto de dados.
history = model.fit(
    dataset_de_treino,
    epochs=NUM_EPOCHS_1,
    steps_per_epoch=len(caminhos_treino) // BATCH_SIZE,
    validation_data=dataset_de_validacao,
    validation_steps=len(caminhos_validacao) // BATCH_SIZE,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
)

#Começando a segunda parte do treinamento do modelo. Nessa parte é realizado um ajuste suave
model.compile(optimizer=Adam(learning_rate=learning_rate_2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    dataset_de_treino,
    epochs=NUM_EPOCHS_2,
    steps_per_epoch=len(caminhos_treino) // BATCH_SIZE,
    validation_data=dataset_de_validacao,
    validation_steps=len(caminhos_validacao) // BATCH_SIZE,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
)

# %%
# Avaliando o modelo carregado
perda, acuracia = model.evaluate(dataset_de_validacao) #Passando o dataset de validacao para o modelo avaliar
print("\nAcuracia: ", np.round(acuracia * 100, 2), '%')
print("\nPerda : ", perda)


''' ----------------------< Avaliação do Modelo >----------------------'''

from tensorflow.keras.models import load_model

# Pegando uma imagem escolhida para avaliação do modelo
test_image = carregaImagem('datasets/' + imagem_teste)
model = load_model('chest_orientation_model.hdf5')
test_image_expd = np.expand_dims(test_image, axis = 0,) / 255
result = model.predict(test_image_expd, batch_size=1)

classe_predita = Le.inverse_transform((np.argmax(result), ))[0]
classe_real = imagem_teste.split('_')[0]

# Previsão da primeira imagem
print('\nMoeda: ', classe_real)
print('\nPredição do Modelo: ', classe_predita)


plt.figure()
plt.plot(history.history['accuracy'], label='Treinamento');plt.plot(history.history['val_accuracy'], label='Validação');
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Acurácia')

plt.figure()
plt.imshow(test_image)
plt.title('Classe: ' + str(classe_real)  + ', Predição: ' + str(classe_predita))


'''----------------------------------------------------------------------'''