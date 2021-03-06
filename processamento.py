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

todos_os_caminhos_original = glob.glob("datasets/*.jpg") # Carrega todas as imagens da pasta raiz e pega todas as imagens dentro da pasta datasets
todos_os_caminhos_original = list(map(lambda x : str(x) , todos_os_caminhos_original)) # Converte todos os objetos da lista para string

shuffle(todos_os_caminhos_original) # Randomiza as imagens

TAMANHO_DA_IMAGEM = 180 #definição do tamanho da imagem.

#Tamanho do lote: define o número de amostras que serão propagadas pela rede. 
#batch_size: Inteiro ou Nenhum. Número de amostras por atualização de gradiente.

BATCH_SIZE = 16    # É definido por causa da performance, para que fique mais fácil e rápido processar todas as imagens. Dentro da pasta Datasets tem-se 3059 imagens, dividida em 5 classe de moedas.
NUM_EPOCHS_1 = 20 # Quantidade de épocas para a primeira parte do treinameto
NUM_EPOCHS_2 = 50 # Quantidade de épocas para a segunda parte do treinameto
learning_rate_1 = 0.001 # Taxa de aprendizagem para a primeira parte do treinameto
learning_rate_2 = 0.000001 # Taxa de aprendizagem para a segunda parte do treinameto
imagem_teste = '25_20.jpg'



'''----------------------------------------------------------------------'''
'''' Definição de funções auxiliares  '''

# Pega os labels das variáveis contidas em todos_os_caminhos
def classificar_rotulo(image_path):
    """
        Transforma o caminho da imagem no rótulo da classe/moeda.
    """
    return int(image_path.split("_")[0].split(os.sep)[-1]) #Divide a string do caminho da imagem na '_' e pega o penúltimo index

def carregaImagem(image, tensor=True):
    """
        Carrega a imagem, aplica o preprocessamento com a biblioteca openCV, redimeniona e retorna uma imagem em formato
        de matriz ou Tensor
    """

    img = cv2.imread(image, cv2.IMREAD_COLOR) # Carrega a imagem em formato de matriz.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #  Converte a imagem BGR em uma imagem RGB
    img = preprocessamento(img, tamanho=TAMANHO_DA_IMAGEM)# Recorta a imagem para a região da moeda e a reescala.
    
    if tensor:
        return tf.constant(img, dtype='uint8') # Converte a representação matricial para a Tensor.'''

    return img # retorna imagem representada em Tensor


def pega_dataset(caminhos, rotulos , train = True):
    """
        Cria um objeto dataset do pacote tensorflow com as imagens especificadas em `caminhos`. Cada imagem é carregada
        utilizando a função`carregaImagem`. Também é aplicada o aumento do dataset.
    """


# aumento do dataset -> adicionando imagens rotacionadas na horizontal, vertical, por angulo, com aumento e diminuição do zoom
    aumento_do_dataset_de_treino = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = (-0.3, -0.2))
    ])

    # função usada para criar um objeto de dados do tensorflow (converte o dataset para algo que o tensorflow consegue entender)
    filtro_automatico = tf.data.experimental.AUTOTUNE

    caminho_da_imagem = caminhos #variavel caminhos imagens
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

#Através da função LabelEncoder() do scikit-learn são atribuidos rótulos numéricos a cada categoria de moeda.

Le = LabelEncoder()

todos_os_rotulos = Le.fit_transform(todos_os_rotulos) #transforma a categoria em um número

print('\n', todos_os_rotulos)

print("\nQuantidade de Moeda: ", len(todos_os_rotulos) )

print("\nQuantidade de Classe: ", len(np.unique(todos_os_rotulos)))

[print(f"\nMoeda de {Le.inverse_transform((cl, ))[0]}: {qtd}") for cl, qtd in zip(*np.unique(todos_os_rotulos, return_counts=True))]


#Separar o conjunto de dados em treino e teste, utilizando 30% da base para teste e o restante para treinamento.

caminhos_treino, caminhos_validacao, \
    rotulos_treino, rotulos_validacao = train_test_split(todos_os_caminhos, todos_os_rotulos,
                                                         test_size=0.3) #separa os dados do dataset em treino e validação (teste).

print("\nQuantidade de Moeda Treinada: ", len(caminhos_treino))
print("\nQuantidade de Moedas Validada: ", len(caminhos_validacao))



# Criando o objeto de dataset de treino e verificando se ele está bom
dataset_de_treino = pega_dataset(caminhos_treino, rotulos_treino) #Cria o dataset de treino

image, label = next(iter(dataset_de_treino)) #Pega uma imagem e um label qualquer (o próximo da iteração nesse caso)
print("\nTamanho do Datasets de Treino: ", image.shape) #Pega o objeto de lista de 4 dimensões com 16 imagens de 180x180 nas 3 cores RBG (vermelho, verde, azul)
print("\nQuantidade de Rotulos de Treino: ", label.shape) #Pega o objeto de lista com todos os rótulos

coin_values = Le.inverse_transform(label)#Escreve o valor da moeda
print('\nPrimeira Moeda do Dataset de Treino: ', coin_values[0])

# Mostra primeira imagem do dataset de treinamento
plt.figure()
plt.imshow(image.numpy()[0])
plt.title(f'Primeira imagem treinada - Moeda de: {Le.inverse_transform([label[0].numpy()])[0]}')


dataset_de_validacao = pega_dataset(caminhos_validacao , rotulos_validacao , train = False) #Mostra o tempo de execução

image, label = next(iter(dataset_de_validacao)) #Pega uma moeda do dataset de validação
print('\nTamanho do Datasets de Validação: ', image.shape) #Imprime o tamanho do dataset de imagens (16 imagens, 180x180 tamanho, 3 cores RGB: vermelho, verde, azul)
print('\nQuantidade de Rotulos de Validação: ', label.shape) #Imprime a quantidade de rótulos

coin_values = Le.inverse_transform(label)#Escreve o valor da moeda
print('\nPrimeira Moeda do Dataset de Validação: ', coin_values[0])#Printa o rótulo da moeda

#Plota a primeira imagem para enxergar o dataset de validação
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

#Rede CNN (Rede Neural Convolucional)
#Camada adicionada ao shape de entrada
# Pode ser feito alteração de inicializacao, bias, entre outros -- https://keras.io/layers/convolutional/#conv2d

# Conv2D --> Camada de convolução, utilizada para convolução nos dados processados.
#'Relu' --> Função de ativação, utilizada para treinar mais rápido o modelo.
# Same --> Preenchimento automático. O Tensorflow espalhar o preenchimento uniformemente à esquerda e à direita.
# MaxPooling2D --> Cada imagem tem seu ruído reduzido e é redimensionada para uma dimensão padronizada.
# BatchNormalization --> a normalização da camada de entrada e das camadas ocultas.
# GlobalMaxPooling2D --> calcula um único valor máximo para cada um dos canais de entrada.
# Dense --> Esta camada são previstos os rótulos e está totalmente conectada.
# Softmax --> utilizada para fazer a ativação da rede de classificação.

model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                 input_shape=(TAMANHO_DA_IMAGEM, TAMANHO_DA_IMAGEM, 3))) #Rede Convolucional recebendo a primeira imagem, mas a camada de ativação 'Relu' e o preenchimento automático 'Same'
model.add(MaxPooling2D((3, 3))) #Camada Pooling

model.add(Conv2D(8, (3, 3), activation='relu', padding='same')) #Rede Convolucional recebendo dados da outra rede, mas a camada de ativação 'Relu' e o preenchimento automático 'Same'
model.add(MaxPooling2D(pool_size=(3, 3))) #Camada Pooling

model.add(Conv2D(8, (3, 3), padding='same'))#Camada de convolução e o preenchimento automático
model.add(BatchNormalization())#Camada BatchNormalization

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))#Camada de convolução e o preenchimento automático
model.add(GlobalMaxPooling2D())#Camada GlobalMaxPooling2D
model.add(BatchNormalization())#Camada BatchNormalization

#Rede MLP (rede Perceptron Multi-Camadas. Ela consiste de uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída)
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
# accuracy --> métrica precisão de acerto
model.compile(optimizer= Adam(learning_rate=learning_rate_1), loss='sparse_categorical_crossentropy',metrics=['accuracy'])


''' ----------------------< Treinamento do Modelo >----------------------'''
reduce_LRO = ReduceLROnPlateau(patience=5, factor=0.01, verbose=True) #ReduceLROnPlateau - aplica a redução da taxa de aprendizado quando a metrica para de ser melhorada.
checkpoint = ModelCheckpoint('chest_orientation_model.hdf5', monitor='val_loss', verbose=1, mode='min', save_best_only=True) #ModelCheckPoint - salva o modelo que tiver o melhor loss durante o treinamento.
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=12, mode='min', verbose=1) #EarlyStop - interrompe o treinamento caso a rede pare de aprender.

#Começando a primeira parte do treinamento  do modelo. Neste momento a rede começa a aprender.
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

#Começando a segunda parte do treinamento do modelo. Nessa parte é realizado um ajuste suave.
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
# Avaliando o modelo carregado.
perda, acuracia = model.evaluate(dataset_de_validacao) #Passando o dataset de validacao para o modelo avaliar.
print("\nAcuracia: ", np.round(acuracia * 100, 2), '%')
print("\nPerda : ", perda)

plt.figure()
plt.plot(history.history['accuracy'], label='Treinamento');plt.plot(history.history['val_accuracy'], label='Validação');
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Acurácia')

''' ----------------------< Avaliação do Modelo >----------------------'''

from tensorflow.keras.models import load_model

# Pegando uma imagem escolhida para avaliação do modelo.
test_image = carregaImagem('datasets/' + imagem_teste)
model = load_model('chest_orientation_model.hdf5') #Carrega o modelo salvo.
test_image_expd = np.expand_dims(test_image, axis = 0,) / 255
result = model.predict(test_image_expd, batch_size=1) #Recebemos as respostas da rede.

classe_predita = Le.inverse_transform((np.argmax(result), ))[0]
classe_real = imagem_teste.split('_')[0]

# Previsão da primeira imagem.
print('\nMoeda: ', classe_real)
print('\nPredição do Modelo: ', classe_predita)

plt.figure()
plt.imshow(test_image)
plt.title('Classe: ' + str(classe_real)  + ', Predição: ' + str(classe_predita))


'''----------------------------------------------------------------------'''
