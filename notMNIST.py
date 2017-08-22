import matplotlib.pyplot as plt
import numpy as np
from  scipy import ndimage
import os
import input_data

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
num_classes = 10
num_per_class = 1000
npy_dataset_file = os.path.join('data_pickle','notMNIST_dataset.npy')
npy_label_file = os.path.join('data_pickle','notMNIST_label.npy')

def save_pickle():
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

def load_notMNIST():
    """
    :return: datasets: shape(image_size * image_size, number of examples)  (784, 10000)
             labels: shape(1, number of examples)  (1, 1000)
    """
    with open(npy_dataset_file,'rb') as f1:
        datasets=np.load(f1)    # datasets.shape = (10000, 28, 28)
    with open(npy_label_file,'rb') as f2:
        labels=np.load(f2)    # labels.shape = (10000, 1)

    #shuffle randomization
    datasets, labels = input_data.randomize(datasets.T, labels.T)
        # datasets.shape = (28, 28, 10000), labels.shape = (1, 10000)

    datasets = datasets.reshape((-1,datasets.shape[-1]))    # datasets.shape = (784, 10000)

    labels_onehot = input_data.onehot_encode(labels, num_classes)    # labels_onehot = (10, 10000)

    return datasets, labels


def display_sample(datasets):
    plt.figure('sample')
    sample_per_class=10
    for i in range(num_classes):
        for j in range(sample_per_class):
            image=datasets[num_per_class*i+j]
            plt.subplot(sample_per_class,sample_per_class,sample_per_class*i+j+1)
            plt.imshow(image)
    plt.show()
