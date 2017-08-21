import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    :param z: a scalar or numpy array of any size.
    :return: the sigmoid function of z
    """
    a = 1/(1+np.exp(-z))
    return a

def relu(z):
    """
    :param z: a scalar or numpy array of any size.
    :return: the relu function of z
    """
    a=z*(z>0)
    return a

def softmax(z):
    """
    :param z: a scalar or numpy array of any size.
    :return: the softmax function of z
    """
    e=np.exp(z)
    a=e/np.sum(e,axis=0,keepdims=True)
    assert(a.shape==e.shape)
    return a

def init_params(layer_dims, scale=0.01, init_type='random'):
    """
    :param layer_dims: python list containing the size of each layer
    :param scale: scalar parameter for random initiation
    :param init_type: 'random', 'zeros'
    :return: parameters: python dictionary containing weight matrix and bias vector,
                        {'Wi': layer i weight matrix of shape (layers_dims[i], layers_dims[i-1]),
                        'bi': layer i bias vector of shape (layers_dims[i], 1)}
    """
    parameters={}
    L=len(layer_dims)
    for l in range(1,L):
        # parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * ((2/layer_dims[l-1])**0.5)
        if init_type=='random':
            parameters['W'+str(l)] = scale * np.random.randn(layer_dims[l],layer_dims[l-1])
        elif init_type=='zeros':
            parameters['W'+str(l)] = np.zeros((layer_dims[l],layer_dims[l-1]))
        elif init_type=='he':
            parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * ((2/layer_dims[l-1])**0.5)
        else:
            raise ValueError()

        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
    return parameters


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: activations from previous layer (or input data): shape(size of previous layer, number of examples)
    :param W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)
    :param activation: "sigmoid" or "relu" "tanh"
    :return: cache: python dictionary for computing backward propagation efficiently
                    {'W': , 'b': ,'A_prev': ,
                    'A': shape(size of current layer, number of examples) ,
                    'Z': shape(size of current layer, number of examples) }
    """
    Z = np.dot(W, A_prev) + b

    if activation=='sigmoid':
        A = sigmoid(Z)
    elif activation=='relu':
        A = relu(Z)
    elif activation=='tanh':
        A = np.tanh(Z)
    elif activation=='softmax':
        A = softmax(Z)
    else:
        raise ValueError('The type of nn layer should be sigmoid, relu, tanh or softmax !!!')

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = {'W':W, 'b':b, 'Z':Z, 'A':A, 'A_prev':A_prev}
    return cache


def forword_propagation(X, parameters, layer_dims, layer_types):
    """
    :param X: input data of shape (size of input layer, number of examples)
    :param parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param layer_dims: python list containing the size of each layer including input layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    :return: Yhat: output of the last activation,
             caches: python list storing cache of each layer, shape (size of output layer, number of examples)
                     cache is a python dictionary{'W': , 'b': , 'Z': , 'A': , 'A_prev': }
    """
    L = len(layer_types)
    caches=[]
    A_prev = X
    for l in range(L):
        W = parameters['W'+str(l+1)]
        b = parameters['b'+str(l+1)]
        cache = linear_activation_forward(A_prev, W, b, layer_types[l])
        A_prev = cache['A']
        caches.append(cache)
    Yhat = caches[L-1]['A']
    assert(Yhat.shape==(layer_dims[-1], X.shape[1]))
    return Yhat, caches

def linear_activation_backward(dA, cache, activation, lambd=0.0):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer
    :param dA: post-activation gradient for current layer
    :param cache: python dictionary{'W': , 'b': , 'Z': , 'A': , 'A_prev': }
    :param activation: "sigmoid" or "relu" "tanh"
    :param lambd: regularization parameter
    :return: dA_prev: gradient of the cost with respect to the activation (of the previous layer), same shape as A_prev
             dW: gradient of the cost with respect to W (current layer), same shape as W
             db: gradient of the cost with respect to b (current layer), same shape as b
    """
    A_prev = cache['A_prev']
    A = cache['A']
    W = cache['W']
    m = A.shape[1]
    if activation=='sigmoid':
        dZ = A*(1-A)*dA
    elif activation=='relu':
        dZ = (A>0)*dA
    elif activation=='tanh':
        dZ = (1-A**2)*dA
    elif activation=='softmax':
        """
        pd--partial derivative
        pdA[i,j] / pdZ[i,j] = A[i,j] * (1 - A[i,j]
        pdA[k,j] / pdZ[i,j] = - A[i,j] * A[k,j],    k!=i
        dZ[i,j] = - A[i,j] * Sigma(dA[k,j] * A[k,j]) + A[i,j] * dA[i,j]
        """
        dZ = - A * np.sum(dA*A, axis=0, keepdims=True) + A*dA
    else:
        raise ValueError('The type of nn layer should be sigmoid, relu, tanh or softmax !!!')

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m

    if lambd != 0.0:
        dW = dW + lambd/m * W
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backword_propagation(X, Y, parameters, caches, layer_dims, layer_types, lambd=0):
    """
    :param X: input data of shape (size of input layer, number of examples)
    :param Y: labels vector of shape (size of output layer, number of examples)
    :param parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param caches: python list storing cache of each layer, cache is a python dictionary{'W': , 'b': , 'Z': , 'A': , 'A_prev': }
    :param layer_dims: python array (list) containing the size of each layer including input layer
    :param layer_types:  python list containing the type of each layer: "sigmoid" or "relu" "tanh" "softmax"
    :param lambd: regularization parameter
    :return: grads: python dictionary storing gradients of each layer, {'dWi': ,'dbi': }
    """
    L = len(layer_types)
    grads={}
    A = caches[L-1]['A']
    if layer_types[-1]=='sigmoid':
        dA = - np.divide(Y, A) + np.divide(1 - Y, 1 - A)
    elif layer_types[-1]=='softmax':
        dA = - np.divide(Y,A)
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')

    grads['dA'+str(L)] = dA

    for l in reversed(range(L)):
        cache=caches[l]
        dA_prev, dW, db=linear_activation_backward(grads['dA'+str(l+1)], cache, layer_types[l], lambd)
        grads['dA'+str(l)] = dA_prev
        grads['dW'+str(l+1)] = dW
        grads['db'+str(l+1)] = db
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    :param parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param grads: python dictionary storing gradients of each layer, {'dWi': ,'dbi': }
    :param learning_rate: scalar
    :return: parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    """
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters


def comput_cost(Yhat, Y, parameters, activation, lambd=0.0):
    """
    :param Yhat: prediction labels vector, of shape (output size, number of examples)
    :param Y: labels vector, of shape (output size, number of examples)
    :param parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param activation: "sigmoid" or "relu" "tanh"
    :param lambd: regularization parameter
    :return: value of the (regularized) loss function
    """
    m = Y.shape[1]
    cost = 0
    if activation=='sigmoid':
        cost = -np.sum(Y*np.log(Yhat)+(1-Y)*np.log(1-Yhat))/m
    elif activation=='softmax':
        cost = -np.sum(Y*np.log(Yhat))/m
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')

    cost = np.squeeze(cost)    #make sure cost.ndim=1
    if lambd != 0.0:
        L = len(parameters) // 2
        r=0
        for l in range(L):
            r = r + np.sum(parameters["W" + str(l+1)]**2)
        cost = cost + r *0.5/m
    return cost


def model(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, lambd=0):
    """
    :param X: input data of shape (size of input layer, number of examples)
    :param Y: labels vector  of shape (size of output layer, number of examples)
    :param hidden_layer_dims: python list containing the size of each hidden layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    :param learning_rate:
    :param num_iterations:
    :param lambd: regularization parameter
    :return: parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    """
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    parameters = init_params(layer_dims)
    costs=[]
    for i in range(num_iterations):
        # Forward propagation
        Yhat,caches = forword_propagation(X, parameters, layer_dims, layer_types)

        # Compute cost
        cost =comput_cost(Yhat, Y, parameters, layer_types[-1],lambd)

        # Backward propagation
        grads = backword_propagation(X, Y, parameters, caches, layer_dims, layer_types, lambd)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(X, Y, parameters, hidden_layer_dims, layer_types, threshold=0.5):
    """
    Predict using learned model (parameters)
    :param X: input data of shape (size of input layer, number of examples)
    :param Y: labels vector  of shape (size of output layer, number of examples)
    :return: parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param hidden_layer_dims: python list containing the size of each hidden layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    :param threshold: for logistic regression
    :return: Y_predict , predicted labels vector of shape (size of output layer, number of examples)
    """
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    Yhat, _ = forword_propagation(X, parameters, layer_dims, layer_types)
    if layer_types[-1]=='sigmoid':
        Y_predict = (Yhat>threshold)*1
    elif layer_types[-1]=='softmax':
        Y_predict = np.argmax(Yhat,axis=0)
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')

    return Y_predict

