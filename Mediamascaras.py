from PIL import Image
import numpy
import os



path = 'C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/'
files = os.listdir(path)
dir1 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/1Epocas5steps202batch4/ImgSegmentadasValid/"
dir2 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/2Epocas5steps202batch4/ImgSegmentadasValid/"
dir3 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/3Epocas5steps202batch4/ImgSegmentadasValid/"
dir4 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/4Epocas5steps202batch4/ImgSegmentadasValid/"
dir5 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/5Epocas5steps202batch4/ImgSegmentadasValid/"
dir6 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/6Epocas5steps202batch4/ImgSegmentadasValid/"
dir7 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/7Epocas5steps202batch4/ImgSegmentadasValid/"
dir8 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/8Epocas5steps202batch4/ImgSegmentadasValid/"
dir9 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/9Epocas5steps202batch4/ImgSegmentadasValid/"
dir10 = "C:/Users/Adm/Desktop/image-segmentation-keras-master/Resultados/10Epocas5steps202batch4/ImgSegmentadasValid/"


dirlist =[dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8,dir9,dir10]

def maskBinaria(mask):
    pixel = mask.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixel[i,j]==(207, 248, 132):
                pixel[i,j]=(1,1,1)
            else:
                pixel[i,j]=(0,0,0)
    return pixel


for file in files[0:1]:
    img=Image.open("C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/"+str(file))

    mascarafinal=numpy.zeros((img.size[1],img.size[0],3))

    for dir in dirlist[0:2]:
        mask = Image.open(dir+str(file[:-4])+"cut.jpg")

        pixel = mask.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixel[i,j]==(207, 248, 132):
                    pixel[i,j]=(1,1,1)
                else:
                    pixel[i,j]=(0,0,0)
        mask = numpy.asarray(mask)
        mascarafinal=mascarafinal+mask
print(mascarafinal[:,:,:].max())
    # for q in range(mascarafinal.shape[0]):
    #     for w in range(mascarafinal.shape[1]):
    #         if mascarafinal[q,w]>(0,0,0):
    #             mascarafinal[q,w]=(1,1,1)
    #         else:
    #             mascarafinal[q,w]=(0,0,0)
    #
    # img = numpy.asarray(img)
    # mascarafinal = numpy.asarray(mascarafinal)
    #
    # # mask.show()
    # final = Image.fromarray(img*mascarafinal, 'RGB')
    # final.show()
