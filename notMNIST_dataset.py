import matplotlib.pyplot as plt
import numpy as np
from  scipy import ndimage
import os
import input_data

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
num_class = 10
num_per_class = 1000
npy_dataset_file = os.path.join('data_pickle','notMNIST_dataset.npy')
npy_label_file = os.path.join('data_pickle','notMNIST_label.npy')

def load_dataset():
    root =  'notMNIST_small'
    filefolders = os.listdir(root)
    datasets=np.ndarray(shape=(num_per_class*len(filefolders), image_size, image_size),dtype=np.float32)
    labels= np.ndarray(shape=(num_per_class*len(filefolders), 1))
    for index, filefolder in enumerate(filefolders):
        images = os.listdir(os.path.join(root,filefolder))
        for i in range(num_per_class):
            image_file = os.path.join(root,filefolder,images[i])
            try:
                image_data = (ndimage.imread(image_file)).astype(float)
                image_data = (image_data - pixel_depth/2) / pixel_depth        #Normalization
                datasets[index*num_per_class+i,:,:] = image_data
            except Exception as e:
                 print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        labels[index*num_per_class:(index+1)*num_per_class,:] = index
    print(datasets.shape)
    print(labels.shape)
    with open(npy_dataset_file,'wb') as f1:
        np.save(f1,datasets)
    with open(npy_label_file,'wb') as f2:
        np.save(f2, labels)

def read_pickle():
    """
    return:  datasets of shape(number of examples, image_size, image_size)  (1000,28,28)
            labels of shape(number of examples, 1)  (1000,1)
    """
    with open(npy_dataset_file,'rb') as f1:
        datasets=np.load(f1)
    with open(npy_label_file,'rb') as f2:
        labels=np.load(f2)
    print(datasets.shape)
    print(labels.shape)
    # display_sample(datasets)
    shuffle_datasets,shuffle_labels = input_data.randomize(datasets, labels)
    return shuffle_datasets,shuffle_labels

def display_sample(datasets):
    plt.figure('sample')
    sample_per_class=10
    for i in range(num_class):
        for j in range(sample_per_class):
            image=datasets[num_per_class*i+j]
            plt.subplot(sample_per_class,sample_per_class,sample_per_class*i+j+1)
            plt.imshow(image)
    plt.show()



datasets,labels=read_pickle()
# display_sample(datasets)
