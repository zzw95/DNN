import matplotlib.pyplot as plt
import numpy as np

def show_decision_boundry(X, Y, Y_onehot, predict_function, parameters, hidden_layer_dims, layer_types, h = 0.02, space = 1):
    """
    plot classification boundry
    :param threshold: for logistic regression
    :param X: input data of shape (size of input layer, number of examples)
    :param Y: labels vector  of shape (1, number of examples), not one-hot
    :param Y_onehot: labels vector  of shape (number of classes, number of examples), one-hot
    :param predict_function: nn_model.predict or nn_model_tf.predict
    :param parameters: parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param hidden_layer_dims: python list containing the size of each hidden layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    :param h: meshgrid size
    :param space: space size
    :return: None
    """
    x1_min = X[0, :].min() - space
    x1_max = X[0, :].max() + space
    x2_min = X[1, :].min() - space
    x2_max = X[1, :].max() + space
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))
    X_meshgird = np.c_[x1.ravel(), x2.ravel()]  # flatten and oncatenate along the second axis
    X_meshgird = X_meshgird.T
    Y_meshgird, _ = predict_function(X_meshgird, Y_onehot, parameters, hidden_layer_dims, layer_types)
    Y_meshgird = Y_meshgird.reshape(x1.shape)
    fig = plt.figure()
    plt.contourf(x1, x2, Y_meshgird, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.show()
