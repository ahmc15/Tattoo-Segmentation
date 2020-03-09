from keras_segmentation.models.unet import unet
import os
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import csv
from keras.optimizers import Adadelta


def autotattoo(epochs,steps_epochs,batch_size,rodada):
    imgOutput_path=str(epochs)+'epocas'+str(steps_epochs)+'steps'+str(batch_size)+'batch'
    def createDIR(epochs, steps_epochs,batch_size,rodada):
        parent_dir ="C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/"
        created_dir = str(rodada)+"Epocas"+str(epochs)+"steps"+str(steps_epochs)+"batch"+str(batch_size)

        PathResultados = os.path.join(parent_dir,created_dir)
        os.mkdir(PathResultados)
        print(PathResultados)
        Diretorios=[]
        lista=['/Mascaras/', '/ImgSegmentadas/', '/Métricas/', '/MascarasValid/', '/ImgSegmentadasValid/']
        for i in lista:
            PathFilhas  = PathResultados+i
            os.mkdir(PathFilhas)
            Diretorios.append(PathFilhas)
        return(Diretorios) #cria os diretórios onde serão salvos os resultados
    def SalvarMetricas(dadosmodel,dir):
        acc=[]
        val_acc=[]
        loss=[]
        val_loss=[]
        for i in range(epochs):
            acc.append(hist[i].history['jaccard_distance'])
            val_acc.append(hist[i].history['val_jaccard_distance'])
            val_loss.append(hist[i].history['val_loss'])
            loss.append(hist[i].history['loss'])

        Metricas = [acc, val_acc,loss,val_loss]
        csvFile = open(dir[2]+imgOutput_path+'.csv', 'w')
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
        plt.savefig(dir[2]+'acuracia')
        plt.figure()
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Perda do Modelo')
        plt.ylabel('Perda')
        plt.xlabel('Épocas')
        plt.legend(['Treino', 'Validação'], loc='upper left')
        plt.savefig(dir[2]+'perdas') #salva as figuras da métrica e da perda de treinamento

    Diretorios=createDIR(epochs, steps_epochs,batch_size,rodada)
    print('C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'train/')
    print('C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'trainmask/')
    print('C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'validmask/')
    print('C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'validmask/')

    model = unet(n_classes=2 ,  input_height=416, input_width=608)
    hist=model.train(
        train_images =  'C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'train/',
        train_annotations = 'C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'trainmask/',
        epochs=epochs,
        val_images='C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'validmask/',
        val_annotations='C:/Users/Adm/Desktop/Kfolds/fold'+str(rodada-1)+'/fold'+str(rodada-1)+'validmask/',
        validate = True,
        steps_per_epoch=steps_epochs,
        batch_size=batch_size)

    SalvarMetricas(hist,Diretorios)


#     path = 'C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/'
#     files = os.listdir(path)
#
#     for file in files:
#         out = model.predict_segmentation(
#             inp="C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/"+str(file),
#             out_fname=Diretorios[3]+str(file[:-4])+"out.png")
#         img=Image.open("C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/"+str(file))
#         mask = Image.open(Diretorios[3]+str(file[:-4])+"out.png")
#         pixel = mask.load()
#         for i in range(img.size[0]):
#             for j in range(img.size[1]):
#                 if pixel[i,j]==(207, 248, 132):
#                     pixel[i,j]=(1,1,1)
#                 else:
#                     pixel[i,j]=(0,0,0)
#         img = numpy.asarray(img)
#         mask = numpy.asarray(mask)
#         final = Image.fromarray(img*mask, 'RGB')
#         final.save(Diretorios[4]+str(file[:-4])+"cut.jpg")
# lista=[]
# with open('parametros.csv') as csvfile:
#     ArquivoCSV=csv.reader(csvfile, delimiter=';')
#     for row in ArquivoCSV:
#         lista.append(row[0].split('\t'))
        #lista.append(row)
print(lista[:])
lista = lista[3:4]
for linha in lista:
    epocas = int(linha[0])
    passo = int(linha[1])
    batch = int(linha[2])
    rodada = int(linha[3])
    autotattoo(epocas,passo,batch,rodada)
