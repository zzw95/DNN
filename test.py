import input_data
import nn_model_tf
import numpy as np
import nn_model
import matplotlib.pyplot as plt

# X, Y = input_data.loadHandwritingTrainingData()
# X=X.T
# Y=Y.T
# layer_types=['tanh','relu','sigmoid']
# hidden_layer_dims=[16,8]
# parameters = nn_model_tf.model(X, Y, hidden_layer_dims, layer_types, 0.05, 200, 0)
# Y_predict = nn_model_tf.predict(X, Y, parameters, hidden_layer_dims, layer_types)
# train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
# print('Training accuracy: %f' % train_accuracy)
# X_test, Y_test = input_data.loadHandwritingTestData()
# X_test=X_test.T
# Y_test=Y_test.T
# Y_test_predict = nn_model_tf.predict(X_test,Y_test, parameters, hidden_layer_dims, layer_types)
# test_accuracy = np.sum(Y_test_predict==Y_test) / Y_test.shape[1]
# print('Test accuracy: %f' % test_accuracy)

def mnist_test():
    from tensorflow.examples.tutorials.mnist import  input_data
    mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
    X, Y = mnist.train.next_batch(100)
    X=X.T
    Y=Y.T
    layer_types=['softmax',]
    hidden_layer_dims=None
    parameters = nn_model_tf.model(X, Y, hidden_layer_dims, layer_types,0.3, 2500, 0)
    Y_predict = nn_model_tf.predict(X, Y, parameters, hidden_layer_dims, layer_types)
    Y_labels = np.argmax(Y,axis=0)
    train_accuracy = np.sum(Y_predict==Y_labels) / Y.shape[1]
    print('Training accuracy: %f' % train_accuracy)

def random_test_tf():
    X, Y, Y_onehot=input_data.loadRandomData()
    layer_types=['relu','softmax',]
    hidden_layer_dims=[100,]
    parameters = nn_model_tf.model(X, Y_onehot, hidden_layer_dims, layer_types, 0.01, 1000, 0)
    Y_predict = nn_model_tf.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
    print('Training accuracy: %f' % train_accuracy)
    print(parameters['W1'])
    h = 0.02
    x1_min, x1_max = X[0, :].min() - 1, X[0, :].max() + 1
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))
    Xt = np.c_[x1.ravel(), x2.ravel()]
    Xt = Xt.T
    Ytp = nn_model_tf.predict(Xt, Y_onehot, parameters, hidden_layer_dims, layer_types)
    Ytp=Ytp.reshape(x1.shape)
    fig = plt.figure()
    plt.contourf(x1, x2, Ytp, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.show()

def random_test():
    X, Y, Y_onehot=input_data.loadRandomData()
    layer_types=['softmax',]
    hidden_layer_dims=None
    parameters = nn_model.model(X, Y_onehot, hidden_layer_dims, layer_types, learning_rate=0.00005, num_iterations=1000)
    Y_predict = nn_model.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
    print('Training accuracy: %f' % train_accuracy)

    h = 0.02
    x1_min, x1_max = X[0, :].min() - 1, X[0, :].max() + 1
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))
    Xt = np.c_[x1.ravel(), x2.ravel()]
    Xt = Xt.T
    Ytp = nn_model.predict(Xt, Y_onehot, parameters, hidden_layer_dims, layer_types)
    print(Ytp.shape)
    Ytp=Ytp.reshape(x1.shape)
    fig = plt.figure()
    plt.contourf(x1, x2, Ytp, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.show()

mnist_test()
