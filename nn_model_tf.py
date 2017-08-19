import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_placeholders(layer_dims):
    X = tf.placeholder(dtype=tf.float64, shape=[layer_dims[0], None])
    Y = tf.placeholder(dtype=tf.float64, shape=[layer_dims[-1], None])
    return X,Y

def init_params(layer_dims,scale=0.01):
    parameters={}
    L=len(layer_dims)
    for l in range(1,L):
        parameters['W'+str(l)]= scale * tf.get_variable('W'+str(l), [layer_dims[l],layer_dims[l-1]], dtype=np.float64,initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b'+str(l)] = tf.Variable(tf.zeros([layer_dims[l],1],dtype=np.float64), name='b'+str(l))
    return parameters

def forword_propagation(X, parameters, layer_dims, layer_types):
    L = len(layer_dims)
    A_prev = X
    for l in range(1,L-1):
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        Z = tf.matmul(W, A_prev)
        if layer_types[l-1]=='sigmoid':
            A = tf.nn.sigmoid(Z, 'Layer'+str(l))
        elif layer_types[l-1]=='relu':
            A = tf.nn.relu(Z, 'Layer'+str(l))
        elif layer_types[l-1]=='tanh':
            A = tf.nn.tanh(Z, 'Layer'+str(l))
        elif layer_types[l-1]=='softmax':
            A = tf.nn.softmax(Z,name='Layer'+str(l))
        A_prev = A
    W = parameters['W'+str(L-1)]
    b = parameters['b'+str(L-1)]
    Z = tf.matmul(W, A_prev)
    return Z


def compute_cost(Z, Y, activation):
    if activation=='sigmoid':
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Z, Y))
    elif activation=='softmax':
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Z, Y))
        # cost = -tf.reduce_mean(Y*tf.log(tf.nn.softmax(Z)))
    return cost


def model(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, lambd=0):

    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])

    with tf.name_scope('input'):
        X_, Y_ = create_placeholders(layer_dims)

    with tf.name_scope('variables'):
        parameters = init_params(layer_dims)
        step=tf.Variable(0,dtype=tf.int32,name="step",trainable=False)

    with tf.name_scope('model'):
        Z = forword_propagation(X_, parameters, layer_dims, layer_types)

    with tf.name_scope('train'):
        cost = compute_cost(Z, Y_, layer_types[-1])
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
        for i in range(num_iterations):
            _, c, summaries, global_step= sess.run([train, cost, summary, step], feed_dict={X_:X, Y_:Y})

            # Print the cost every 100 training example
            if i % 10 == 0:
                writer.add_summary(summaries,global_step=global_step)
                print ("Cost after iteration %i: %f" %(i, c))
                costs.append(c)

        params = sess.run(parameters)
        writer.flush()
        writer.close()

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return params

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

    X_, Y_ = create_placeholders(layer_dims)

    Z = forword_propagation(X_, parameters, layer_dims, layer_types)

    if layer_types[-1]=='sigmoid':
        A = tf.sigmoid(Z)
    elif layer_types[-1]=='softmax':
        A = tf.nn.softmax(Z)

    with tf.Session() as sess:
        Yhat=A.eval(feed_dict={X_:X, Y_:Y}, session=sess)

    if layer_types[-1]=='sigmoid':
        Y_prediction = (Yhat>threshold)*1
    elif layer_types[-1]=='softmax':
        Y_prediction = np.argmax(Yhat,axis=0)

    return Y_prediction




