import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_placeholders(layer_dims):
    """
    :param layer_dims: python list containing the size of each layer including input layer
    :return:  X: placeholder for the data input, of shape [size of input layer, None] and dtype "float"
              Y: placeholder for the input labels, of shape [size of output layer, None] and dtype "float"
    """
    X = tf.placeholder(dtype=tf.float32, shape=[layer_dims[0], None])
    Y = tf.placeholder(dtype=tf.float32, shape=[layer_dims[-1], None])
    return X,Y

def init_params(layer_dims, scale=0.01):
    """
    :param layer_dims: python list containing the size of each layer including input layer
    :param scale: scalar
    :return: parameters: python dictionary of variable tensors containing weight and bias of each layer,
                        {'Wi': layer i weight tensor of shape (layers_dims[i], layers_dims[i-1]),
                        'bi': layer i bias tensor of shape (layers_dims[i], 1)}
    """
    parameters={}
    L=len(layer_dims)
    for l in range(1,L):
        # parameters['W'+str(l)] = tf.Variable(tf.truncated_normal(shape=[layer_dims[l],layer_dims[l-1]]),name='W'+str(l))
        # parameters['W'+str(l)]= scale * tf.get_variable('W'+str(l), [layer_dims[l],layer_dims[l-1]], dtype=np.float64,initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['W'+str(l)] = tf.Variable(np.random.randn(layer_dims[l], layer_dims[l-1]) *scale,dtype=np.float32, name='W'+str(l))
        parameters['b'+str(l)] = tf.Variable(tf.zeros([layer_dims[l],1],dtype=np.float32), name='b'+str(l))
    return parameters

def forword_propagation(X, parameters, layer_dims, layer_types, prob = 1):
    """
    :param X: the data input tensor
    :param parameters: python dictionary containing weight and bias tensor, {'Wi': , 'bi': }
    :param layer_dims: python list containing the size of each layer including input layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh" "softmax"
    :return: Z_output: the output tensor of the last LINEAR unit
    """
    L = len(layer_dims)
    A_prev = X
    for l in range(1,L-1):
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        Z = tf.add(tf.matmul(W, A_prev), b)
        if layer_types[l-1]=='sigmoid':
            A = tf.nn.sigmoid(Z, name='Layer'+str(l))
        elif layer_types[l-1]=='relu':
            A = tf.nn.relu(Z, name='Layer'+str(l))
        elif layer_types[l-1]=='tanh':
            A = tf.nn.tanh(Z, name='Layer'+str(l))
        elif layer_types[l-1]=='softmax':
            A = tf.nn.softmax(Z,name='Layer'+str(l), dim=0)
            # Be caraful, the default feature dim in tensorflow is -1, so here the dim must be changed

        if(prob<1):
            A = tf.nn.dropout(A, keep_prob=prob)
        A_prev = A
    W = parameters['W'+str(L-1)]
    b = parameters['b'+str(L-1)]
    Z_output = tf.add(tf.matmul(W, A_prev), b)
    return Z_output


def compute_cost(Z, Y, parameters, activation, lambd):
    """
    :param Z: the output tensor of the last LINEAR unit
    :param Y: the labels tensor
    :param parameters: python dictionary containing weight and bias tensor, {'Wi': , 'bi': }
    :param activation: the activation function type of output layer, "sigmoid" "softmax"
    :param lambd: L2 regularization parameter, scalar
    :return:
    """
    L = len(parameters) // 2
    if activation=='sigmoid':
        if lambd != 0:
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Z, Y) +
                                  tf.add_n( [tf.nn.l2_loss(parameters["W" + str(l+1)]) for l in range(L)] )* lambd)
        else:
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Z, Y))

    elif activation=='softmax':
        if lambd != 0:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Z, Y, dim =0) +
                                  tf.add_n( [tf.nn.l2_loss(parameters["W" + str(l+1)]) for l in range(L)] )* lambd)
        else:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Z, Y, dim =0))
        # Be caraful, the default feature dim in tensorflow is -1, so here the dim must be changed
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')


    return cost


def model(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, num_batches=1, lambd=0, prob =1):
    """
    :param X: the data input matrix of shape (size of input layer, number of examples)
    :param Y: the labels vector  of shape (size of output layer, number of examples)
    :param hidden_layer_dims: python list containing the size of each hidden layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh" "softmax"
    :param learning_rate: scalar
    :param num_iterations: scalar
    :param num_batches: scalar
    :param lambd: scalar, regularization parameter
    :return: params: python dictionary containing weight and bias tensor of each layer, {'Wi': , 'bi': }
    """
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])

    with tf.name_scope('input'):
        X_, Y_ = create_placeholders(layer_dims)

    with tf.name_scope('variables'):
        parameters = init_params(layer_dims)
        step = tf.Variable(0,dtype=tf.int32,name="step",trainable=False)

    with tf.name_scope('model'):
        Z = forword_propagation(X_, parameters, layer_dims, layer_types, prob)

    with tf.name_scope('train'):
        cost = compute_cost(Z, Y_, parameters, layer_types[-1], lambd)
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=step)

    with tf.name_scope('summary'):
        tf.summary.scalar('cost',cost)

    with tf.name_scope('global_ops'):
        init = tf.global_variables_initializer()
        summary=tf.summary.merge_all()

    costs=[]

    with tf.Session() as sess:
        sess.run(init)
        writer=tf.summary.FileWriter('./graph',sess.graph)
        batch_size = Y.shape[1] // num_batches
        for i in range(num_iterations):
            for j in range(num_batches):
                _, c, summaries, global_step = sess.run([train, cost, summary, step],
                                                       feed_dict={X_:X[:, j*batch_size : (j+1)*batch_size],
                                                                  Y_:Y[:, j*batch_size : (j+1)*batch_size]})

                # Print the cost every 100 training example
                if i % 10 == 0:
                    writer.add_summary(summaries,global_step=global_step)
                    print ("Cost after iteration %d and batch %d: %f" %(i, j, c))
                    costs.append(c)

        params = sess.run(parameters)
        writer.flush()
        writer.close()

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    return params

def predict(X, Y, parameters, hidden_layer_dims, layer_types, threshold=0.5):
    """
    :param X: the data input matrix of shape (size of input layer, number of examples)
    :param Y: the labels vector  of shape (size of output layer, number of examples)
    :param parameters: python dictionary containing weight and bias tensor of each layer, {'Wi': , 'bi': }
    :param hidden_layer_dims: python list containing the size of each hidden layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh" "softmax"
    :param threshold: scalr for sigmoid
    :return: Y_predict: predicted labels vector of shape (1, number of examples)
             accuracy: scalar
    """
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])

    X_, Y_ = create_placeholders(layer_dims)

    Z = forword_propagation(X_, parameters, layer_dims, layer_types)

    if layer_types[-1]=='sigmoid':
        A = tf.nn.sigmoid(Z)
    elif layer_types[-1]=='softmax':
        A = tf.nn.softmax(Z, dim=0)
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')

    with tf.Session() as sess:
        Yhat=A.eval(feed_dict={X_:X, Y_:Y}, session=sess)

    if layer_types[-1]=='sigmoid':
        Y_predict = (Yhat>threshold)*1
        accuracy = np.sum(Y_predict ==Y) / Y.shape[1]
    elif layer_types[-1]=='softmax':
        Y_predict = np.argmax(Yhat,axis=0).reshape((1,-1))
        accuracy = np.sum(np.squeeze(Y_predict) == np.argmax(Y, axis=0)) / Y.shape[1]

    return Y_predict, accuracy




