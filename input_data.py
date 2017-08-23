import numpy as np
import os
import matplotlib.pyplot as plt

def img2vec(fileName):
    digitsList=[]
    with open(fileName) as f:       #32*32
        lines = f.readlines()
        for line in lines:
            line=line.strip()
            for c in line:
                digitsList.append(c)
    digits=np.array(digitsList)
    return digits

def loadHandwritingTrainingData():
    """
    :return: trainDigits: shape(1024, 387), number of examples = 387
             trainLabels: shape(1024,1)
    """
    trainRootDir='digits\\trainingDigits'
    trainFileList=os.listdir(trainRootDir)
    trainFileNum = len(trainFileList)
    trainDigits = np.zeros((trainFileNum,1024))
    trainLabels = np.zeros((trainFileNum,1))
    for i in range(trainFileNum):
        fileNameStr= trainFileList[i]
        trainLabels[i,:] = fileNameStr.split('_')[0]
        trainDigits[i,:] = img2vec(os.path.join(trainRootDir,fileNameStr))

    trainDigits = trainDigits.T
    trainLabels = trainLabels.T

    trainDigits, trainLabels = randomize(trainDigits, trainLabels)
    assert(trainDigits.shape[1]==trainLabels.shape[1])
    print(trainDigits.shape)
    return trainDigits, trainLabels

def loadHandwritingTestData():
    """
    :return: testDigits: shape(1024, 184), number of examples = 184
             testLabels: shape(1, 184)
    """
    testRootDir='digits\\testDigits'
    testFileList=os.listdir(testRootDir)
    testFileNum = len(testFileList)
    testDigits = np.zeros((testFileNum,1024))
    testLabels = np.zeros((testFileNum,1))
    for i in range(testFileNum):
        fileNameStr= testFileList[i]
        testLabels[i,:] = fileNameStr.split('_')[0]
        testDigits[i,:] = img2vec(os.path.join(testRootDir,fileNameStr))

    testDigits = testDigits.T
    testLabels = testLabels.T
    assert(testDigits.shape[1]==testLabels.shape[1])

    testDigits, testLabels = randomize(testDigits, testLabels)
    assert(testDigits.shape[1]==testLabels.shape[1])

    return testDigits, testLabels

def loadRandomData(N=100,D=2,K=3):
    """
    :param N: number of points per class
    :param D: dimensionality
    :param K: number of classes
    :return: X: data matrix of shape(D, N*K), number of examples = N*K
             Y: labels vector of shape(1, N*K)
             Y_onehot: one-hot labels matrix of shape(K, N*K)
    """
    X = np.zeros([N*K,D]) # data matrix (each row = single example)
    Y = np.zeros([N*K,1],dtype=int) # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    X, Y =randomize(X.T, Y.T)
    assert(X.shape[1]==Y.shape[1])
    Y_onehot = onehot_encode(Y,K)
    return X,Y, Y_onehot



def onehot_encode(Y, C):
    """
    :param: Y -- vector containing the labels, shape = (1,number of examples) or (number of examples)
    :param: C -- number of classes, the depth of the one hot dimension
    :return: Y_onehot -- one hot matrix, shape = (C, number of examples)
    """
    Y_onehot = (np.arange(C)[:,None]  == Y[None,:]).astype(np.float32)    #broadcasting
    Y_onehot = np.squeeze(Y_onehot)
    assert(Y_onehot.shape==(C, Y.shape[1]))
    return Y_onehot

def randomize(datasets, labels):
    """
    :param datasets: ndarray, shape(None, None, number of examples) or (None, number of examples)
    :param labels: ndarray, shape(None,number of examples) or (number of examples,)
    :return: shuffle_dataset, shuffle_labels
    """
    permutation = np.random.permutation(datasets.shape[-1])
    if datasets.ndim==3:
        shuffle_dataset=datasets[:,:,permutation]
    elif datasets.ndim==2:
        shuffle_dataset=datasets[:,permutation]
    else:
        shuffle_dataset=None

    if labels.ndim==2:
        shuffle_labels = labels[:,permutation]
    elif labels.ndim==1:
        shuffle_labels = labels[permutation]
    else:
        shuffle_labels = None

    return shuffle_dataset,shuffle_labels
