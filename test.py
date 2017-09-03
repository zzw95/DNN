import input_data
import nn_model_tf
import numpy as np
import nn_model
import matplotlib.pyplot as plt
import notMNIST_MNIST
import plot


def handWriting_test():
    X_train, Y_train = input_data.loadHandwritingTrainingData()    #X_train.shape=(1024,387)  Y_train.shape=(1,387)
    layer_types= ['sigmoid']
    hidden_layer_dims=None
    parameters = nn_model.model(X_train, Y_train, hidden_layer_dims, layer_types, learning_rate=0.1, num_iterations= 501)

    Y_train_predict, train_accuracy  = nn_model.predict(X_train, Y_train, parameters, hidden_layer_dims, layer_types)
    print('Training accuracy: %f' % train_accuracy)

    X_test, Y_test = input_data.loadHandwritingTestData()    #X_test.shape=(1024, 184)  Y_train.shape=(1, 184)
    Y_test_predict, test_accuracy = nn_model.predict(X_test, Y_test, parameters, hidden_layer_dims, layer_types)
    print(Y_test[0,:10])
    print(Y_test_predict[0,:10])
    print('Test accuracy: %f' % test_accuracy)


def random_test():
    X, Y, Y_onehot=input_data.loadRandomData()
    layer_types=['relu','softmax',]
    hidden_layer_dims=[120,]
    parameters = nn_model.model(X, Y_onehot, hidden_layer_dims, layer_types, learning_rate=0.5, num_iterations=2001, lambd=1.2)
    Y_predict, train_accuracy = nn_model.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
    print('Training accuracy: %f' % train_accuracy)

    plot.show_decision_boundry(X, Y, Y_onehot, nn_model.predict, parameters, hidden_layer_dims, layer_types)


def random_test_with_dropout():
    X, Y, Y_onehot=input_data.loadRandomData()
    layer_types=['relu','softmax',]
    hidden_layer_dims=[120,]
    parameters = nn_model.model_with_dropout(X, Y_onehot, hidden_layer_dims, layer_types, learning_rate=0.5, num_iterations=2001, num_batches=2)
    Y_predict, train_accuracy = nn_model.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
    print('Training accuracy: %f' % train_accuracy)

    plot.show_decision_boundry(X, Y, Y_onehot, nn_model.predict, parameters, hidden_layer_dims, layer_types)

def random_test_with_adam():
    X, Y, Y_onehot=input_data.loadRandomData()
    layer_types=['relu','softmax',]
    hidden_layer_dims=[120,]
    parameters = nn_model.model_with_adam(X, Y_onehot, hidden_layer_dims, layer_types, learning_rate=0.5, num_iterations=2001, num_batches=2)
    Y_predict, train_accuracy = nn_model.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
    print('Training accuracy: %f' % train_accuracy)

    plot.show_decision_boundry(X, Y, Y_onehot, nn_model.predict, parameters, hidden_layer_dims, layer_types)

def random_test_with_dropout_adam():
    X, Y, Y_onehot=input_data.loadRandomData()
    layer_types=['relu','softmax',]
    hidden_layer_dims=[120,]
    parameters = nn_model.model_with_dropout_adam(X, Y_onehot, hidden_layer_dims, layer_types, learning_rate=0.5, num_iterations=2001)
    Y_predict, train_accuracy = nn_model.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    train_accuracy = np.sum(Y_predict==Y) / Y.shape[1]
    print('Training accuracy: %f' % train_accuracy)

    # plot.show_decision_boundry(X, Y, Y_onehot, nn_model.predict, parameters, hidden_layer_dims, layer_types)


def not_mnist_test():
    X, Y, Y_onehot = notMNIST_MNIST.load_notMNIST()
        # X.shape = (784, 10000), Y.shape = (1, 10000), Y_onehot.shape = (10, 10000)
    layer_types=['relu','softmax',]
    hidden_layer_dims=[100,]
    parameters = nn_model.model(X, Y_onehot, hidden_layer_dims, layer_types, learning_rate=0.8, num_iterations=101, num_batches = 10)
    Y_predict, train_accuracy = nn_model.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    print('Training accuracy: %f' % train_accuracy)
    image_shape=(28,28)
    sample_indices = [0,10,100,200,500,1000,2000,5000,9000]
    alphabet_list = np.array([chr(i) for i in range(ord('A'), ord('J')+1)])
    print(alphabet_list[ list( map(int ,Y[0,sample_indices]) ) ])
    print(alphabet_list[ Y_predict[0, sample_indices] ])
    plot.display_image_samples(X, image_shape, sample_indices)


def mnist_test():
    X, Y, Y_onehot = notMNIST_MNIST.load_MINIST(10000)
        # X.shape = (784, 10000), Y.shape = (1, 10000), Y_onehot.shape = (10, 10000)
    layer_types=['softmax',]
    hidden_layer_dims=None
    parameters = nn_model.model(X, Y_onehot, hidden_layer_dims, layer_types, learning_rate=0.5, num_iterations=101, num_batches = 10)
    Y_predict, train_accuracy = nn_model.predict(X, Y_onehot, parameters, hidden_layer_dims, layer_types)
    print('Training accuracy: %f' % train_accuracy)
    image_shape = (28,28)
    sample_indices = [0,10,100,200,500,1000,2000,5000,9000]
    print(Y[0,sample_indices])
    print(Y_predict[0,sample_indices])
    plot.display_image_samples(X, image_shape, sample_indices)

# handWriting_test()
# random_test()
random_test_with_dropout()
# random_test_with_adam()
# random_test_with_dropout_adam()
# not_mnist_test()
# mnist_test()
