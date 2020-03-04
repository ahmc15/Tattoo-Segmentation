import os
import random
import shutil



pathMask = 'C:/Users/Adm/Desktop/Alltattoomasks/'
def moverMacaras(contador,pathMask):
    pathTattoo = 'C:/Users/Adm/Desktop/Kfolds/fold'+str(contador)+'/fold'+str(contador)+'valid/'
    alvo       = pathTattoo[:-1]+'mask/'


    mascaras = os.listdir(pathMask)
    tatuagens = os.listdir(pathTattoo)


    for mask in mascaras:
        for tattoo in tatuagens:
            if mask[:-4] == tattoo[:-4]:
                shutil.copy(pathMask+mask, alvo)
            else:
                pass

for i in range(0,10):
    moverMacaras(i,pathMask)
