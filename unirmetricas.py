
import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
# def listaNomeCSV101():
#     listaNomes=[]
#     for i in range(1,11):
#         listaNomes.append('C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/ResultadosArquiteturaMetade/ResultadosEpocas100steps101batch8/'+str(i)+'Epocas100steps101batch8/Métricas/100epocas101steps8batch.csv')
#     return listaNomes
def listaNomeCSV202():
    listaNomes=[]
    for i in range(1,11):
        listaNomes.append('C:/Users/Adm/Desktop/Tattoo-Segmentation/Resultados/lr10-5momentum0,99batch8/'+str(i)+'Epocas200steps101batch8/Metricas/200epocas101steps8batch.csv')
                            #C:\Users\Adm\Desktop\Tattoo-Segmentation\Resultados\lr10-5momentum0,99batch8
    return listaNomes
# def listaNomeCSV404():
#     listaNomes=[]
#     for i in range(1,11):
#         listaNomes.append('C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/ResultadosArquiteturaMetade/ResultadosEpocas100steps404batch2/'+str(i)+'Epocas100steps404batch2/Métricas/100epocas404steps2batch.csv')
#     return listaNomes
# def listaNomeCSV801():
#     listaNomes=[]
#     for i in range(1,11):
#         listaNomes.append('C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/'+str(i)+'Epocas200steps801batch1/Métricas/200epocas801steps1batch.csv')
#     return listaNomes


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
    STDIoU=[]
    IoU_val=[]
    STDIoU_val=[]
    loss=[]
    STDloss=[]
    loss_val=[]
    STDloss_val=[]

    for count, item in enumerate(lista1[0]):
        IoU.append(sum([lista1[0][count],lista2[0][count],lista3[0][count],lista4[0][count],lista5[0][count],lista6[0][count],lista7[0][count],lista8[0][count],lista9[0][count],lista10[0][count]])/10)
        STDIoU.append(np.std(np.array([lista1[0][count],lista2[0][count],lista3[0][count],lista4[0][count],lista5[0][count],lista6[0][count],lista7[0][count],lista8[0][count],lista9[0][count],lista10[0][count]])))

        IoU_val.append(sum([lista1[1][count],lista2[1][count],lista3[1][count],lista4[1][count],lista5[1][count],lista6[1][count],lista7[1][count],lista8[1][count],lista9[1][count],lista10[1][count]])/10)
        STDIoU_val.append(np.std(np.array([lista1[1][count],lista2[1][count],lista3[1][count],lista4[1][count],lista5[1][count],lista6[1][count],lista7[1][count],lista8[1][count],lista9[1][count],lista10[1][count]])))

        loss.append(sum([lista1[2][count],lista2[2][count],lista3[2][count],lista4[2][count],lista5[2][count],lista6[2][count],lista7[2][count],lista8[2][count],lista9[2][count],lista10[2][count]])/10)
        STDloss.append(np.std(np.array([lista1[2][count],lista2[2][count],lista3[2][count],lista4[2][count],lista5[2][count],lista6[2][count],lista7[2][count],lista8[2][count],lista9[2][count],lista10[2][count]])))

        loss_val.append(sum([lista1[3][count],lista2[3][count],lista3[3][count],lista4[3][count],lista5[3][count],lista6[3][count],lista7[3][count],lista8[3][count],lista9[3][count],lista10[3][count]])/10)
        STDloss_val.append(np.std(np.array([lista1[3][count],lista2[3][count],lista3[3][count],lista4[3][count],lista5[3][count],lista6[3][count],lista7[3][count],lista8[3][count],lista9[3][count],lista10[3][count]])))

    return [IoU, STDIoU, IoU_val, STDIoU_val, loss, STDloss, loss_val, STDloss_val]

# listinha=listaNomeCSV101()
X = MediaKfolds(listaNomeCSV202())
# # Y = MediaKfolds(listaNomeCSV404())
# # W = MediaKfolds(listaNomeCSV801())
# # Z = MediaKfolds(listaNomeCSV101())
x = np.linspace(0, 1, 200)
plt.figure(figsize=(10,8), dpi=120)
# # plt.plot(Y[0])
# # plt.plot(Y[1])
# #plt.plot(X[0])
plt.errorbar(x, X[0], X[1])
plt.errorbar(x, X[2], X[3])
# # plt.plot(W[0])
# # plt.plot(W[1])
# # plt.plot(Z[0])
# # plt.plot(Z[1])
plt.title('jaccard do Modelo')

plt.ylabel('IoU')
plt.xlabel('Épocas')
plt.legend(['TreinoBatch8', 'ValidaçãoBatch8','TreinoBatch2', 'ValidaçãoBatch2','TreinoBatch1', 'ValidaçãoBatch1','TreinoBatch8', 'ValidaçãoBatch8'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
# plt.show()
plt.savefig('C:/Users/Adm/Desktop/Tattoo-Segmentation/Resultados/lr10-5momentum0,99batch8/iouArqoriginal_200_101_8error.png')
plt.figure(figsize=(10,8))
plt.errorbar(x, X[4], X[5])
plt.errorbar(x, X[6], X[7])
# # plt.plot(Y[2])
# # plt.plot(Y[3])
# # plt.plot(W[2])
# # plt.plot(W[3])
# # plt.plot(Z[2])
# # plt.plot(Z[3])
plt.title('Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Épocas')
plt.legend(['TreinoBatch8', 'ValidaçãoBatch8','TreinoBatch2', 'ValidaçãoBatch2','TreinoBatch1', 'ValidaçãoBatch1','TreinoBatch8', 'ValidaçãoBatch8'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
# plt.show()
plt.savefig('C:/Users/Adm/Desktop/Tattoo-Segmentation/Resultados/lr10-5momentum0,99batch8/perdasArqoriginal_200_101_8error.png')
