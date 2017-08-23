import input_data
import nn_model_tf
import numpy as np
import nn_model
import matplotlib.pyplot as plt
import notMNIST_MNIST

def gradient_check(X, Y, layer_dims, layer_types, parameters,lambd, epsilon = 1e-5, num_params =8):
    """
    check the difference of gradients dW between backward propagation and numerical approxiation
    :param X: input data of shape (size of input layer, number of examples)
    :param Y: labels vector of shape (size of output layer, number of examples)
    :param layer_dims: python array (list) containing the size of each layer including input layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    :param parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param lambd: L2 regularization parameter
    :param epsilon: scalar
    :param num_params: scalar
    :return: None
    """
    L = len(layer_dims)
    grad = np.zeros((num_params, L-1))
    grad_approx = np.zeros((num_params, L-1))
    for l in range(1,L):
        print('========This is %dst %s layer========' % (l, layer_types[l-1]))
        count = 0
        offset = layer_dims[l-1] // (num_params-1)
        check_params = range(0, layer_dims[l-1], offset)
        for i in check_params:
            print('----This is %dst param----' % i)
            Wl = parameters['W'+str(l)]
            print(Wl.shape)
            print("W%d[0][%d] = %f" % (l, i, Wl[0,i]))
            Wl_plus = Wl[0][i] + epsilon
            Wl_minus = Wl[0][i] - epsilon
            Yhat,caches = nn_model.forword_propagation(X, parameters, layer_dims, layer_types)
            J = nn_model.comput_cost(Yhat, Y, parameters, layer_types[-1], lambd)
            grads = nn_model.backword_propagation(X, Y, parameters, caches, layer_dims, layer_types, lambd)
            dWl = grads['dW'+str(l)]
            print(np.sum(dWl>0))
            grad[count,l-1] = dWl[0,i]
            print('dW%d[0][%d] = %f' %(l, i,dWl[0,i]))
            print('J = '+str(J))

            parameters['W'+str(l)][0][i] = Wl_plus
            print('W%d_plus[0][%d] = %f' % (l, i, parameters['W'+str(l)][0,i]))
            Yhat,caches = nn_model.forword_propagation(X, parameters, layer_dims, layer_types)
            J_plus = nn_model.comput_cost(Yhat, Y, parameters, layer_types[-1], lambd)
            print('J_plus = '+str(J_plus))

            parameters['W'+str(l)][0][i] = Wl_minus
            print('W%d_minus[%d][0] = %f' % (l, i, parameters['W'+str(l)][0,i]))
            Yhat,caches = nn_model.forword_propagation(X, parameters, layer_dims, layer_types)
            J_minus = nn_model.comput_cost(Yhat, Y, parameters, layer_types[-1], lambd)
            print('J_minus = '+str(J_minus))

            dWl_0i = (J_plus - J_minus) * 0.5 / epsilon
            grad_approx[count,l-1] = dWl_0i
            print('dW%d[0][%d]_approx = %f' % (l, i, dWl_0i))

            count = count + 1

    numerator = np.linalg.norm(grad-grad_approx)
    denominator = np.linalg.norm(grad)+np.linalg.norm(grad_approx)
    if denominator==0:
        difference = 0.0
    else:
        difference = numerator/denominator
    print("********differrence = "+str(difference))


def check_sigmoid():
    X, Y = input_data.loadHandwritingTrainingData()    #X.shape=(1024,387)  Y.shape=(1,387)
    layer_types= ['relu','tanh','sigmoid']
    layer_dims=[X.shape[0],100,50, Y.shape[0]]
    parameters = nn_model.init_params(layer_dims)
    gradient_check(X, Y, layer_dims, layer_types, parameters, lambd =1)

def check_softmax():
    X,Y,Y_onehot = input_data.loadRandomData()
    # X.shape=(2, 300), Y.shape=(1,300), Y_onehot=(3,300)
    # number of examples = 300, number of classes = 3

    layer_types= ['softmax',]
    layer_dims=[X.shape[0],  Y_onehot.shape[0]]
    parameters = nn_model.init_params(layer_dims)
    gradient_check(X, Y_onehot, layer_dims, layer_types, parameters,epsilon = 1e-7, num_params =2, lambd = 1)


def gradient_check_with_dorpout(X, Y, layer_dims, layer_types, parameters, epsilon = 1e-5, num_params =8, keep_prob =0.5):
    """
    check the difference of gradients dW between backward propagation and numerical approxiation
    :param X: input data of shape (size of input layer, number of examples)
    :param Y: labels vector of shape (size of output layer, number of examples)
    :param layer_dims: python array (list) containing the size of each layer including input layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    :param parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    :param epsilon: scalar
    :param num_params: scalar
    :param: dropout parameter
    :return: None
    """
    L = len(layer_dims)
    grad = np.zeros((num_params, L-1))
    grad_approx = np.zeros((num_params, L-1))
    for l in range(1,L):
        print('========This is %dst %s layer========' % (l, layer_types[l-1]))
        count = 0
        offset = layer_dims[l-1] // (num_params-1)
        check_params = range(0, layer_dims[l-1], offset)
        for i in check_params:
            print('----This is %dst param----' % i)
            Wl = parameters['W'+str(l)]
            print(Wl.shape)
            print("W%d[0][%d] = %.10f" % (l, i, Wl[0,i]))
            Wl_plus = Wl[0][i] + epsilon
            Wl_minus = Wl[0][i] - epsilon
            Yhat,caches = nn_model.forword_propagation_with_dropout(X, parameters, layer_dims, layer_types, keep_prob)
            J = nn_model.comput_cost(Yhat, Y, parameters, layer_types[-1], lambd=0)
            grads = nn_model.backword_propagation_with_dropout(X, Y, parameters, caches, layer_dims, layer_types, keep_prob)
            dWl = grads['dW'+str(l)]
            print(np.sum(dWl>0))
            grad[count,l-1] = dWl[0,i]
            print('dW%d[0][%d] = %.10f' %(l, i,dWl[0,i]))
            print('J = '+str(J))

            parameters['W'+str(l)][0][i] = Wl_plus
            print('W%d_plus[0][%d] = %.10f' % (l, i, parameters['W'+str(l)][0,i]))
            Yhat,caches = nn_model.forword_propagation(X, parameters, layer_dims, layer_types)
            J_plus = nn_model.comput_cost(Yhat, Y, parameters, layer_types[-1], lambd=0)
            print('J_plus = '+str(J_plus))

            parameters['W'+str(l)][0][i] = Wl_minus
            print('W%d_minus[%d][0] = %.10f' % (l, i, parameters['W'+str(l)][0,i]))
            Yhat,caches = nn_model.forword_propagation(X, parameters, layer_dims, layer_types)
            J_minus = nn_model.comput_cost(Yhat, Y, parameters, layer_types[-1], lambd=0)
            print('J_minus = '+str(J_minus))

            dWl_0i = (J_plus - J_minus) * 0.5 / epsilon
            grad_approx[count,l-1] = dWl_0i
            print('dW%d[0][%d]_approx = %.10f' % (l, i, dWl_0i))

            count = count + 1
    print(grad)
    print(grad_approx)
    numerator = np.linalg.norm(grad-grad_approx)
    denominator = np.linalg.norm(grad)+np.linalg.norm(grad_approx)
    if denominator== 0:
        difference = 0.0
    else:
        difference = numerator/denominator
    print("********differrence = "+str(difference))


def check_dropout():
    X,Y,Y_onehot = input_data.loadRandomData()
    layer_types= ['softmax',]
    layer_dims=[X.shape[0], Y_onehot.shape[0]]
    parameters = nn_model.init_params(layer_dims)
    gradient_check_with_dorpout(X, Y, layer_dims, layer_types, parameters,num_params =2)


check_dropout()
