from PIL import Image
import numpy

img=Image.open("C:/Users/Adm/Desktop/TattooSegmentation/test_frames/test/0.jpg")
mask = Image.open('out1.png')

pixel = mask.load()
print(pixel[0,0])
print(pixel[150,250])
for i in range(img.size[0]):
    for j in range(img.size[1]):
        if pixel[i,j]==(207, 248, 132):
            pixel[i,j]=(1,1,1)
        else:
            pixel[i,j]=(0,0,0)

img = numpy.asarray(img)
mask = numpy.asarray(mask)

# mask.show()
final = Image.fromarray(img*mask, 'RGB')
final.show()
