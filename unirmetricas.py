
import csv
from PIL import Image
from numpy import mean
import matplotlib.pyplot as plt

def IOUfromCSV(fileString):
    lista=[]
    myFile = open(fileString, 'r')
    ArquivoCSV=csv.reader(myFile, delimiter=';')
    for row in ArquivoCSV:
        lista.append(row[0].split('/t'))

    IoU = lista[0][0].split(',')
    IoU_val = lista[1][0].split(',')
    loss = lista[2][0].split(',')
    loss_val = lista[3][0].split(',')

    for i in range(len(IoU)):
        IoU[i] = float(IoU[i][1:-1])
        IoU_val[i] = float(IoU_val[i][1:-1])
        loss[i] = float(loss[i][1:-1])
        loss_val[i] = float(loss_val[i][1:-1])

    return [IoU,IoU_val,loss,loss_val]
def listaNomeCSV101():
    listaNomes=[]
    for i in range(1,11):
        listaNomes.append('C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/ResultadosArquiteturaMetade/ResultadosEpocas100steps101batch8/'+str(i)+'Epocas100steps101batch8/Métricas/100epocas101steps8batch.csv')
    return listaNomes
def listaNomeCSV202():
    listaNomes=[]
    for i in range(1,4):
        listaNomes.append('C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/learningrate0,001/'+str(i)+'Epocas200steps202batch4/Métricas/200epocas202steps4batch.csv')
    return listaNomes
def listaNomeCSV404():
    listaNomes=[]
    for i in range(1,11):
        listaNomes.append('C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/ResultadosArquiteturaMetade/ResultadosEpocas100steps404batch2/'+str(i)+'Epocas100steps404batch2/Métricas/100epocas404steps2batch.csv')
    return listaNomes
def listaNomeCSV808():
    listaNomes=[]
    for i in range(1,11):
        listaNomes.append('C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/ResultadosArquiteturaMetade/ResultadosEpocas100steps808batch1/'+str(i)+'Epocas100steps808batch1/Métricas/100epocas808steps1batch.csv')
    return listaNomes


def MediaKfolds(listaNomes):
    lista1 = IOUfromCSV(listaNomes[0])
    lista2 = IOUfromCSV(listaNomes[1])
    lista3 = IOUfromCSV(listaNomes[2])
    lista4 = IOUfromCSV(listaNomes[3])
    lista5 = IOUfromCSV(listaNomes[4])
    lista6 = IOUfromCSV(listaNomes[5])
    lista7 = IOUfromCSV(listaNomes[6])
    lista8 = IOUfromCSV(listaNomes[7])
    lista9 = IOUfromCSV(listaNomes[8])
    lista10 = IOUfromCSV(listaNomes[9])
    IoU=[]
    IoU_val=[]
    loss=[]
    loss_val=[]

    for i in range(len(lista1[0])):
        IoU.append(sum([lista1[0][i],lista2[0][i],lista3[0][i],lista4[0][i],lista5[0][i],lista6[0][i],lista7[0][i],lista8[0][i],lista9[0][i],lista10[0][i]])/10)

        IoU_val.append(sum([lista1[1][i],lista2[1][i],lista3[1][i],lista4[1][i],lista5[1][i],lista6[1][i],lista7[1][i],lista8[1][i],lista9[1][i],lista10[1][i]])/10)

        loss.append(sum([lista1[2][i],lista2[2][i],lista3[2][i],lista4[2][i],lista5[2][i],lista6[2][i],lista7[2][i],lista8[2][i],lista9[2][i],lista10[2][i]])/10)

        loss_val.append(sum([lista1[3][i],lista2[3][i],lista3[3][i],lista4[3][i],lista5[3][i],lista6[3][i],lista7[3][i],lista8[3][i],lista9[3][i],lista10[3][i]])/10)
    return [IoU,IoU_val,loss,loss_val]

listinha=listaNomeCSV101()
X = MediaKfolds(listaNomeCSV202())
# Y = MediaKfolds(listaNomeCSV404())
# W = MediaKfolds(listaNomeCSV808())
# Z = MediaKfolds(listaNomeCSV101())

plt.figure(figsize=(10,8), dpi=120)
# plt.plot(Y[0])
# plt.plot(Y[1])
plt.plot(X[0])
plt.plot(X[1])
# plt.plot(W[0])
# plt.plot(W[1])
# plt.plot(Z[0])
# plt.plot(Z[1])
plt.title('jaccard do Modelo')

plt.ylabel('IoU')
plt.xlabel('Épocas')
# plt.legend(['Treino202', 'Validação202','Treino404', 'Validação404','Treino808', 'Validação808'], loc='upper left')
plt.legend(['TreinoBatch4', 'ValidaçãoBatch4','TreinoBatch2', 'ValidaçãoBatch2','TreinoBatch1', 'ValidaçãoBatch1','TreinoBatch8', 'ValidaçãoBatch8'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
# plt.show()
plt.savefig('iouArq200')
plt.figure(figsize=(10,8))
plt.plot(X[2])
plt.plot(X[3])
# plt.plot(Y[2])
# plt.plot(Y[3])
# plt.plot(W[2])
# plt.plot(W[3])
# plt.plot(Z[2])
# plt.plot(Z[3])
plt.title('Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Épocas')
plt.legend(['TreinoBatch4', 'ValidaçãoBatch4','TreinoBatch2', 'ValidaçãoBatch2','TreinoBatch1', 'ValidaçãoBatch1','TreinoBatch8', 'ValidaçãoBatch8'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.savefig('perdasArq200')
