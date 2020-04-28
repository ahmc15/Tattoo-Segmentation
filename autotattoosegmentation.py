#!/opt/conda/bin python3


from keras_segmentation.models.unet import unet
from keras_segmentation.models.segnet import vgg_segnet

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
#config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import os
import math
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import csv
from keras.optimizers import Adadelta
from keras.optimizers import SGD


def createDIR(epochs, steps_epochs,batch_size,rodada):
    '''
    Função utilizada para criar o diretório no qual serão salvos os resultados
    de cada treinamento.

    A função cria o diretório "created_dir" dentro do diretório "parent_dir" já
    existente. O diretório "created_dir" terá as subpastas contendo os diferentes
    produtos do treinamento de segmentador. As pastas contendo o resultados são
    criadas à partir da lista "subpasta".

    #Subpastas
        Mascaras: as máscaras da imagens segmentadas pelo modelo
        ImgSegmentadas: as imagens de teste segmentadas
        Métricas: as métricas de treinamento do modelo

    '''
    #parent_dir ="C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/"
    parent_dir ="/mnt/nas/AndreCosta/Tattoo-Segmentation/Resultados/"
    created_dir = str(rodada)+"Epocas"+str(epochs)+"steps"+str(steps_epochs)+"batch"+str(batch_size)

    PathResultados = os.path.join(parent_dir,created_dir)
    os.mkdir(PathResultados)
    Diretorios=[]
    subpasta=['/Mascaras/', '/ImgSegmentadas/', '/Metricas/']
    for i in subpasta:
        PathFilhas  = PathResultados+i
        os.mkdir(PathFilhas)
        Diretorios.append(PathFilhas)
    return(Diretorios) #cria os diretórios onde serão salvos os resultados

def SalvarMetricas(dadosmodel,Diretorios,imgOutput_path,epochs):
    '''
    Função que exporta a métrica e a perda de treinamento e de validação para o
    terceiro diretório da Lista "Diretorios".

    Salva em listas os dados presentes no dicionário "history". As métricas são
    unidas em uma lista nomeada "Metricas" que é exportada em um arquivo CSV.
    Além disso, salva os gráficos da Métrica pelas Épocas e da Perda pelas Épocas
    em formato png.
    '''
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for i in range(epochs):
        acc.append(dadosmodel[i].history['jaccard_distance'])
        val_acc.append(dadosmodel[i].history['val_jaccard_distance'])
        val_loss.append(dadosmodel[i].history['val_loss'])
        loss.append(dadosmodel[i].history['loss'])

    Metricas = [acc, val_acc,loss,val_loss]
    csvFile = open(Diretorios[2]+imgOutput_path+'.csv', 'w')
    with csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerows(Metricas)
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Jaccard do Modelo')
    plt.ylabel('Jaccard Distance')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig(Diretorios[2]+'metrica')
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig(Diretorios[2]+'perdas') #salva as figuras da métrica e da perda de treinamento

def predictTattoo(pathImg,Diretorios,modelo):
    '''
    Função que faz a predição das máscaras e faz a exportação das imagens
    sobrepostas com as respectivas máscaras.

    A função cria uma lista com as imagens presentes no diretório "pathImg". Para
    cada imagem, é feito a predição de uma máscara da tatuagem que é salva em
    formato png. Em seguida, são lidas os pares de imagens e máscaras. è feita
    uma trasnformação no valor dos pixels da máscara para poder multiplicar as
    imagens e as máscaras. Dessa forma, apenas o conteúdo predito como tatuagem
    é mostrado na imagem segmentada. A imagem segmentada é exportada em formato
    JPEG.
    '''
    files = os.listdir(pathImg)
    for file in files:
        out = modelo.predict_segmentation(
            inp=pathImg+str(file),
            out_fname=Diretorios[0]+str(file[:-4])+"out.png")
        img=Image.open(pathImg+str(file))
        mask = Image.open(Diretorios[0]+str(file[:-4])+"out.png")
        pixel = mask.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixel[i,j]==(207, 248, 132):
                    pixel[i,j]=(1,1,1)
                else:
                    pixel[i,j]=(0,0,0)
        img = numpy.asarray(img)
        mask = numpy.asarray(mask)
        final = Image.fromarray(img*mask, 'RGB')
        final.save(Diretorios[1]+str(file[:-4])+"cut.jpg")

def autotattoo(epochs,batch_size,rodada):
    '''
    Função de treinamento de um segmentador. Recebe como input:
    # Argumentos
        epochs: número inteiro que determina o número de épocas de treinamento
        batch_size = número inteiro que determina o tamanho do batch de treinamento
        rodada = número intero que determina a pasta do k-fold a ser utilizada

    O passo da janela dos batches é feito pelo arredondamento para cima da
    divisão do número de imagens de treinamento(801) pelo tamanho de cada batch.

    imgOutput_path é uma string utilizada para criar o nome das pastas com os
    resultados do treinamento.

    PathKfolds: diretório onde se encontram os K-folds.

    O modelo é selecionado pela função U-NET. Seu treinamento é feito a partir
    da função "train" que recebe os diretórios contendo as imagens e máscaras de
    treinamento e validação, os parâmetros de treinamento e otimizador da função
    de custo,"optimizer_tattoo".

    Função SalvarMetricas exporta a métrica e a perda de treinamento.

    pathImgTeste: Diretório onde se encontram as imagens de teste a serem segmentadas.

    Função predictTattoo segmenta as imagens de teste.
    '''
    steps_epochs = math.ceil(801/batch_size)
    imgOutput_path=str(epochs)+'epocas'+str(steps_epochs)+'steps'+str(batch_size)+'batch'
    #PathKfolds = 'C:/Users/Adm/Desktop/Kfolds/'
    PathKfolds = '/mnt/nas/AndreCosta/Kfolds/'
    Diretorios = createDIR(epochs, steps_epochs,batch_size,rodada)

    optimizer_tattoo = SGD(lr=0.00001, momentum=0.99, nesterov=False)
    model = unet(n_classes=2, input_height=416, input_width=608)
    hist=model.train(
        train_images =  PathKfolds+'fold'+str(rodada-1)+'/fold'+str(rodada-1)+'train/',
        train_annotations = PathKfolds+'fold'+str(rodada-1)+'/fold'+str(rodada-1)+'trainmask/',
        epochs=epochs,
        val_images = PathKfolds+'fold'+str(rodada-1)+'/fold'+str(rodada-1)+'validmask/',
        val_annotations = PathKfolds+'fold'+str(rodada-1)+'/fold'+str(rodada-1)+'validmask/',
        validate = True,
        steps_per_epoch=steps_epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_tattoo)

    SalvarMetricas(hist,Diretorios,imgOutput_path,epochs)
    # pathImgTeste = 'C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/'
    # predictTattoo(pathImgTeste,Diretorios,model)



def main():
    '''
    Função que automatiza o processo de treinamento do segmentador com os
    diferentes k-folds. Os parâmetros são feitos pela lista <Parametros>.
    A lista é populada pelas linhas do arquivo <parametros.csv>. Cada linha do
    CSV é composta por uma string com diferentes valores separados por espaço e
    duas linhas são delimitada por ";".

    Após a criação da lista de parâmetros, é eliminada o primeiro elemento da
    lista que contem apenas o cabeçalho do CSV.

    A lista "Parametros" faz a correspondência com os folds da seguinte forma:
    #Exemplo:
        Parametros[1:] faz o treinamento com o primeiro fold em diante.
        ...
        Parametros[5:7] faz o treinamento com o quinto e sexto fold.
    '''
    Parametros=[]
    with open('parametros.csv') as csvfile:
        ArquivoCSV=csv.reader(csvfile, delimiter=';')
        for row in ArquivoCSV:
            Parametros.append(row[0].split('\t'))

    Parametros = Parametros[1:]
    for linha in Parametros:
        epocas = int(linha[0])
        batch = int(linha[2])
        rodada = int(linha[3])
        autotattoo(epocas,batch,rodada)

if __name__ == "__main__":
    main()
