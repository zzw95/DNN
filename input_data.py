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
    trainRootDir='digits\\trainingDigits'
    trainFileList=os.listdir(trainRootDir)
    trainFileNum = len(trainFileList)
    trainDigits = np.zeros((trainFileNum,1024))
    trainLabels = np.zeros((trainFileNum,1))
    for i in range(trainFileNum):
        fileNameStr= trainFileList[i]
        trainLabels[i,:] = fileNameStr.split('_')[0]
        trainDigits[i,:] = img2vec(os.path.join(trainRootDir,fileNameStr))
    return trainDigits, trainLabels

def loadHandwritingTestData():
    testRootDir='digits\\testDigits'
    testFileList=os.listdir(testRootDir)
    testFileNum = len(testFileList)
    testDigits = np.zeros((testFileNum,1024))
    testLabels = np.zeros((testFileNum,1))
    for i in range(testFileNum):
        fileNameStr= testFileList[i]
        testLabels[i,:] = fileNameStr.split('_')[0]
        testDigits[i,:] = img2vec(os.path.join(testRootDir,fileNameStr))
    return testDigits, testLabels

def loadRandomData(N=100,D=2,K=3):
    """
    :param N: number of points per class
    :param D: dimensionality
    :param K: number of classes
    :return:
    """
    X = np.zeros([N*K,D]) # data matrix (each row = single example)
    Y = np.zeros([N*K,1],dtype=int) # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    Y_onehot = np.zeros((K, N*K))
    Y_onehot[Y.squeeze(),:] = 1
    return X.T,Y.T, Y_onehot

