import os
# path = 'C:/Users/Adm/Desktop/train_frames/train/'
path = 'C:/Users/Adm/Desktop/TattooSegmentation/val_frames/val'
files = os.listdir(path)
print(files[0])
for file in files:
    os.rename(os.path.join(path,file), os.path.join(path,file[3:]))
