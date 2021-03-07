# Tif -> Numpy
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# DATA PATH
data_path='./datasets/'

filename_label='train-labels.tif'
filename_train='train-volume.tif'
filename_test='test-volume.tif' # 나중에 사용

img_label=Image.open(os.path.join(data_path,filename_label))
img_train=Image.open(os.path.join(data_path,filename_train))

y_size, x_size = img_label.size
nframe=img_label.n_frames

#print(nframe) 30개
train_count=24
vali_count=6

id_frame=np.arange(nframe)
np.random.shuffle(id_frame)

def save_npy(file_path,img_data,idx_range):
    count=0
    id_frame, image, label = img_data
    start, end = idx_range
    for idx in range(start,end):
        label.seek(id_frame[idx])
        image.seek(id_frame[idx])
        label_ = np.asarray(label)
        image_ = np.asarray(image)
        label_path=file_path+('label_%03d.npy'%count)
        image_path=file_path+('image_%03d.npy'%count)
        np.save(label_path,label_)
        np.save(image_path,image_)
        count+=1

img_data=[id_frame,img_train,img_label]
save_npy(data_path+'train/',img_data,(0,train_count))
save_npy(data_path+'vali/',img_data,(train_count,nframe))