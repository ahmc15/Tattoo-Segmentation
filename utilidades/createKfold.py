import os
import random
import shutil

random.seed(4)
path='C:/Users/Adm/Desktop/ALLtattooframes/'
path2='C:/Users/Adm/Desktop/Kfolds/'
files = os.listdir(path)
random.shuffle(files)
print(len(files))
partes=[]
for i in range(10):
    nomedir = path2+'fold'+str(i)
    os.mkdir(nomedir)
    unidade = []
    unidade =  list(files[(i*89):((i+1)*89)])
    #partes.append(unidade)
    for file in range(len(unidade)):
        shutil.copy(path+unidade[file], nomedir)
