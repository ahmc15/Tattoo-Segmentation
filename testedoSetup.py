from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
import matplotlib.pyplot as plt

model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

#model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

#model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="C:/Users/Adm/Desktop/image-segmentation-keras-master/sample_images/1_input.jpg",
    out_fname="out.png"
)
plt.imshow(out)
