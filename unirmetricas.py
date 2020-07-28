import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def IOUfromCSV(fileString):
    '''
    Função que retira do arquivo CSV a métrica e as perdas do processo de
    treinamento e validação.
    #1
        lê cada linha do arquivo CSV e separa o primeiro elemento de cada linha
        como o elemento(string) de uma lista.
    #2
        Separa o elemento de cada lista nas vírgulas em diferentes string.
    #3
        elimina os colchetes que acompanham os números e transforma os strings
        em float.
    #RETORNA
        Uma lista contendo 4 listas, sendo cada uma, uma das medidas.
    '''
    lista=[]
    myFile = open(fileString, 'r')
    ArquivoCSV=csv.reader(myFile, delimiter=';')
    #1
    for row in ArquivoCSV:
        lista.append(row[0].split('/t'))
    #2
    IoU = lista[0][0].split(',')
    IoU_val = lista[1][0].split(',')
    loss = lista[2][0].split(',')
    loss_val = lista[3][0].split(',')
    #3
    for count, item in enumerate(IoU):
        IoU[count] = float(IoU[count][1:-1])
        IoU_val[count] = float(IoU_val[count][1:-1])
        loss[count] = float(loss[count][1:-1])
        loss_val[count] = float(loss_val[count][1:-1])

    return [IoU,IoU_val,loss,loss_val]

def listaNomeCSV(path):
    '''
    Função encontra todos os arquivos csv dentro dos sub diretórios de um
    detereminado diretório.
    Retorna uma lista com as paths dos arquivos CSV.
    '''
    listaNomes=[]
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".csv")):
                listaNomes.append(os.path.join(root,name))
    return listaNomes

def MediaKfolds(listaNomes):
    '''
    Retorna uma lista com as médias e os desvios padrões das métricas e perdas.
    '''
    Lista_CSV_Dados = list(map(IOUfromCSV,listaNomes))
    for folds,item in enumerate(Lista_CSV_Dados):
        if folds == 0:
            iou_array=np.array(item[0])
            iouval_array=np.array(item[1])
            loss_array=np.array(item[2])
            lossval_array=np.array(item[3])
        else:
            iou_array    =np.vstack((iou_array    ,np.array(item[0])))
            iouval_array =np.vstack((iouval_array ,np.array(item[1])))
            loss_array   =np.vstack((loss_array   ,np.array(item[2])))
            lossval_array=np.vstack((lossval_array,np.array(item[3])))
    IoU        =np.mean(iou_array,axis=0)
    STDIoU     =np.std(iou_array,axis=0)
    IoU_val    =np.mean(iouval_array,axis=0)
    STDIoU_val =np.std(iouval_array,axis=0)
    loss       =np.mean(loss_array,axis=0)
    STDloss    =np.std(loss_array,axis=0)
    loss_val   =np.mean(lossval_array,axis=0)
    STDloss_val=np.std(lossval_array,axis=0)

    return [IoU, STDIoU, IoU_val, STDIoU_val, loss, STDloss, loss_val, STDloss_val]



savepath = 'C:/Users/Adm/Desktop/Tattoo-Segmentation/Resultados/lr10-5momentum0,9batch8epocas1000/'
X = MediaKfolds(listaNomeCSV(savepath))
x = np.linspace(0, 1, 1000)
plt.figure(figsize=(10,8), dpi=120)
plt.plot(X[0])
plt.plot(X[2])
plt.title('jaccard do Modelo')
plt.ylim(0.1, 1.2)
plt.ylabel('IoU')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Validação'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.savefig(savepath+'iou_1000_101_8aug.png')
plt.figure(figsize=(10,8))
plt.plot(X[4])
plt.plot(X[6])
plt.title('Perda do Modelo')
plt.ylim(0.1, 1.2)
plt.ylabel('Perda')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Validação'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.savefig(savepath+'perdas_1000_101_8aug.png')

'''Gráfico com barras de erro '''

plt.figure(figsize=(10,8), dpi=120)
plt.errorbar(x, X[0], X[1])
plt.errorbar(x, X[2], X[3])
plt.title('jaccard do Modelo')
plt.ylim(0.1, 1.2)
plt.ylabel('IoU')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Validação'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.savefig(savepath+'ERRORiou_1000_101_8aug.png')
plt.figure(figsize=(10,8))
plt.errorbar(x, X[4], X[5])
plt.errorbar(x, X[6], X[7])
plt.title('Perda do Modelo')
plt.ylim(0.1, 1.2)
plt.ylabel('Perda')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Validação'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.savefig(savepath+'ERRORperdas_1000_101_8aug.png')
