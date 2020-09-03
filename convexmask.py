from skimage import io
import numpy
from skimage.morphology import convex_hull_image

def mask_convexa_dir(path_mask, path_img):
    '''
    Calcula a casca convexa de uma máscara resultado da rede de segmentação

    A função implementa o algoritmo da casca convexa, criando o menor polígono
    convexo que englobe todos os pontos da máscara.
    #Argumentos
        path_mask: endereço do arquivo da máscara
        path_img: endereço do arquivo da imagem

    #Resultado
        img: imagem resultado da sobreposição do máscara convexa e da imagem original

    '''
    mask = io.imread(path_mask)
    img = io.imread(path_img)
    mask_bin = numpy.zeros((mask.shape[0],mask.shape[1]))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if numpy.all(mask[i,j,:]==[207, 248, 132]):
                mask_bin[i,j]=True
            else:
                mask_bin[i,j]=False
    mask_bin = convex_hull_image(mask_bin)
    mask_bin = numpy.dstack((mask_bin,mask_bin,mask_bin))
    img = img*mask_bin
    return img
def mask_convexa(img_mask, img_tattoo):
    '''
    Calcula a casca convexa de uma máscara resultado da rede de segmentação

    A função implementa o algoritmo da casca convexa, criando o menor polígono
    convexo que englobe todos os pontos da máscara.
    #Argumentos
        img_mask: numpy array da imagem da máscara
        img_tattoo: numpy array da imagem da tatuagem

    #Resultado
        img: imagem resultado da sobreposição do máscara convexa e da imagem original

    '''
    mask = img_mask
    img = img_tattoo
    mask_bin = numpy.zeros((mask.shape[0],mask.shape[1]))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if numpy.all(mask[i,j,:]==[207, 248, 132]):
                mask_bin[i,j]=True
            else:
                mask_bin[i,j]=False
    mask_bin = convex_hull_image(mask_bin)
    mask_bin = numpy.dstack((mask_bin,mask_bin,mask_bin))
    img = img*mask_bin
    return img

name = 'pabloortiz81'
path = 'D:/Resultados/5Epocas1000steps101batch8/Mascaras/'+name+'out.png'
img = 'C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/'+name+'.jpg'
path = io.imread(path)
img = io.imread(img)
result =  mask_convexa(path,img)
io.imshow(result)
io.show()
