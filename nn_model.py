import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- the sigmoid function of z
    """
    s=1/(1+np.exp(-z))
    return s

def relu(z):
    """
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- the relu function of z
    """
    s=z*(z>0)
    return s

def softmax(z):
    """
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- the softmax function of z
    """
    e=np.exp(z)
    s=e/np.sum(e,axis=-1,keepdims=True)
    return s

def init_params(layer_dims,scale=0.01):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters={}
    L=len(layer_dims)
    for l in range(1,L):
        # parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * ((2/layer_dims[l-1])**0.5)
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * scale
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" "tanh"
    Returns:
    cache -- stored for computing the backward pass efficiently
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
    cache = {'W':W, 'b':b, 'Z':Z, 'A':A, 'A_prev':A_prev}
    return cache

def linear_activation_backward(dA, cache, activation, lambd=0):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- python dictionary store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" "tanh"
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
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
        dZ = A *(1-A)*dA
    dW = np.dot(dZ, A_prev.T)/m + lambd/m * W
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward
    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters

def forword_propagation(X, parameters, layer_dims, layer_types):
    """
    Implement the backward propagation
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    layer_dims -- python array (list) containing the size of each layer.
    layer_types --python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    Returns:
    Yhat -- The sigmoid output of the last activation
    caches -- python list containning cache
    """
    L = len(layer_dims)
    caches=[]
    A_prev = X
    for l in range(1,L):
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        cache = linear_activation_forward(A_prev, W, b, layer_types[l-1])
        A_prev = cache['A']
        caches.append(cache)
    Yhat = caches[L-2]['A']
    return Yhat, caches

def backword_propagation(X, Y, parameters, caches, layer_dims, layer_types, lambd=0):
    """
    Implement the backward propagation
    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    layer_dims -- python array (list) containing the size of each layer.
    layer_types --python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    L = len(layer_dims)
    grads={}
    A = caches[L-2]['A']
    if layer_types[-1]=='sigmoid':
        dA = - np.divide(Y, A) + np.divide(1 - Y, 1 - A)
    elif layer_types[-1]=='softmax':
        dA = - np.divide(Y,A)

    for l in reversed(range(1,L)):
        cache=caches[l-1]
        dA_prev, dW, db=linear_activation_backward(dA, cache, layer_types[l-1], lambd)
        grads['dA'+str(l-1)] = dA_prev
        grads['dW'+str(l)] = dW
        grads['db'+str(l)] = db
        dA = dA_prev
    return grads

def comput_cost(Yhat, Y, parameters, activation, lambd=0.0):
    """
    Implement the cost function with L2 regularization.
    Arguments:
    Yhat -- prediction labels vector, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    cost = 0
    if activation=='sigmoid':
        cost = -np.sum(Y*np.log(Yhat)+(1-Y)*np.log(1-Yhat))/m
    elif activation=='softmax':
        cost = -np.sum(Y*np.log(Yhat))/m
    r=0
    L = len(parameters) // 2
    if lambd != 0.0:
        for l in range(L):
            r = r + np.sum(parameters["W" + str(l+1)]**2)
    cost_with_regularization= cost + r *0.5/m
    return cost_with_regularization


def model(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, lambd=0):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    num_iterations -- Number of iterations in gradient descent loop
    hidden_layer_dims -- python array (list) containing the size of each hidden layer
    layer_types --python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
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
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    Arguments:
    parameters -- parameters learnt by the model. They can then be used to predict.
    X -- data of size (num_px * num_px * 3, number of examples)
    hidden_layer_dims -- python array (list) containing the size of each hidden layer
    layer_types --python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    Yhat, _ = forword_propagation(X, parameters, layer_dims, layer_types)
    if layer_types[-1]=='sigmoid':
        Y_prediction = (Yhat>threshold)*1
    elif layer_types[-1]=='softmax':
        Y_prediction = np.argmax(Yhat,axis=0)
    return Y_prediction


# import input_data
# X, Y = input_data.loadHandwritingTrainingData()
# X=X.T
# Y=Y.T
# layer_types=['tanh','relu','sigmoid']
# hidden_layer_dims=[16,8]
# parameters = model(X, Y, hidden_layer_dims, layer_types, 0.1, 2000, 0)
# Y_predict = predict(X, parameters, hidden_layer_dims, layer_types)
# train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
# print('Training accuracy: %f' % train_accuracy)
# X_test, Y_test = input_data.loadHandwritingTestData()
# X_test=X_test.T
# Y_test=Y_test.T
# Y_test_predict = predict(X_test, parameters, hidden_layer_dims, layer_types)
# test_accuracy = np.sum(Y_test_predict==Y_test) / Y_test.shape[1]
# print('Test accuracy: %f' % test_accuracy)


# import input_data
# X, Y = input_data.loadHandwritingTrainingData()
# X=X.T
# Y=Y.T
# layer_types=['sigmoid']
# hidden_layer_dims=None
# parameters = model(X, Y, hidden_layer_dims, layer_types, 0.1, 2000, 0)
# Y_predict = predict(X, Y, parameters, hidden_layer_dims, layer_types)
# train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
# print('Training accuracy: %f' % train_accuracy)
# X_test, Y_test = input_data.loadHandwritingTestData()
# X_test=X_test.T
# Y_test=Y_test.T
# Y_test_predict = predict(X_test, parameters, hidden_layer_dims, layer_types)
# test_accuracy = np.sum(Y_test_predict==Y_test) / Y_test.shape[1]
# print('Test accuracy: %f' % test_accuracy)
