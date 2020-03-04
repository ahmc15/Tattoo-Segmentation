from keras_segmentation.models.unet import unet
import os
from PIL import Image
import numpy
import matplotlib.pyplot as plt


epochs=5
steps_epochs=808
batch_size=1
rodada = 25
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
    return(Diretorios)
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
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('jaccard Acurácia do Modelo')
    plt.ylabel('Acurácia')
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
    plt.savefig(dir[2]+'perdas')

Diretorios=createDIR(epochs, steps_epochs,batch_size,rodada)

model = unet(n_classes=2 ,  input_height=416, input_width=608)
hist=model.train(
    train_images =  'C:/Users/Adm/Desktop/Kfolds/fold0/fold0train/',
    train_annotations = 'C:/Users/Adm/Desktop/Kfolds/fold0/fold0trainmask/',
    epochs=epochs,
    val_images='C:/Users/Adm/Desktop/Kfolds/fold0/fold0validmask/',
    val_annotations='C:/Users/Adm/Desktop/Kfolds/fold0/fold0validmask/',
    validate = True,
    steps_per_epoch=steps_epochs,
    batch_size=batch_size)


SalvarMetricas(hist,Diretorios)


path = 'C:/Users/Adm/Desktop/TattooSegmentation/test_frames/teste00/'
files = os.listdir(path)
# 'C:/Users/Adm/Desktop/TattooSegmentation/test_frames/teste00/'
for file in files:
    out = model.predict_segmentation(
        inp=path+str(file),
        out_fname=Diretorios[0]+str(file[:-4])+"out.png")
    img=Image.open("C:/Users/Adm/Desktop/TattooSegmentation/test_frames/teste00/"+str(file))
    mask = Image.open(Diretorios[0]+str(file[:-4])+"out.png")
    pixel = mask.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixel[i,j]==(207, 248, 132):
                pixel[i,j]=(1,1,1)
            else:
                pixel[i,j]=(0,0,0)
    img = numpy.asarray(img)
    mask = numpy.asarray(mask)
    final = Image.fromarray(img*mask, 'RGB')
    final.save(Diretorios[1]+str(file[:-4])+"cut.jpg")

path = 'C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/'
files = os.listdir(path)

for file in files:
    out = model.predict_segmentation(
        inp="C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/"+str(file),
        out_fname=Diretorios[3]+str(file[:-4])+"out.png")
    img=Image.open("C:/Users/Adm/Desktop/TattooSegmentation/test_frames/valid/"+str(file))
    mask = Image.open(Diretorios[3]+str(file[:-4])+"out.png")
    pixel = mask.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixel[i,j]==(207, 248, 132):
                pixel[i,j]=(1,1,1)
            else:
                pixel[i,j]=(0,0,0)
    img = numpy.asarray(img)
    mask = numpy.asarray(mask)
    final = Image.fromarray(img*mask, 'RGB')
    final.save(Diretorios[4]+str(file[:-4])+"cut.jpg")

# #import matplotlib.pyplot as plt
# #plt.imshow(out)

# evaluating the model
# print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
