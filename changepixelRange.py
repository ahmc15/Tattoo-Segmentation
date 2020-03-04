from PIL import Image
import os

path = 'C:/Users/Adm/Desktop/TattooSegmentation/val_masks/val/'
files = os.listdir(path)

for file in files:
    img = Image.open(path+file)
    # img = Image.open('C:/Users/Adm/Desktop/image-segmentation-keras-master/testedecor3.png')
    pixel=img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixel[i,j]>=(127,127,127):
                pixel[i,j]=(1,1,1)
            else:
                pixel[i,j]=(0,0,0)
    # img.save('testedecor3.png')
    img.save('C:/Users/Adm/Desktop/train_masksPNG/valZERO/'+file[:-4]+'.png')
