"""Iris Classifier."""
import lasagne as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
import theano
import theano.tensor as T
import time


def report(text, output_file):
    f = open(output_file, 'a')
    f.write('{}\n'.format(text))
    f.close


def load_iris_data(filename='iris.txt'):
    with open(filename, 'rb') as f:
        X = [line.split()[0].split(',') for line in f]

    tag = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    Y = [tag[x[4]] for x in X]
    X0, X1, X2 = [], [], []

    for x in X:
        if x[4] == 'Iris-setosa':
            X0.append([float(x[0]), float(x[1]), float(x[2]), float(x[3])])
        elif x[4] == 'Iris-versicolor':
            X1.append([float(x[0]), float(x[1]), float(x[2]), float(x[3])])
        elif x[4] == 'Iris-virginica':
            X2.append([float(x[0]), float(x[1]), float(x[2]), float(x[3])])

    X0 = np.array(X0).astype('float32')
    X1 = np.array(X1).astype('float32')
    X2 = np.array(X2).astype('float32')

    np.random.shuffle(X0)
    np.random.shuffle(X1)
    np.random.shuffle(X2)

    X_train = np.concatenate([X0[:25], X1[:25], X2[:25]])
    Y_train = np.concatenate([[0]*25, [1]*25, [2]*25]).astype('int32')
    X_test = np.concatenate([X0[25:], X1[25:], X2[25:]])
    Y_test = np.concatenate([[0]*25, [1]*25, [2]*25]).astype('int32')

    return X_train, Y_train, X_test, Y_test


def generate_minibatches(data, targets, batch_size=25, stochastic=True):
    assert len(data) == len(targets)
    idx = np.arange(len(data))
    if stochastic:
        np.random.shuffle(idx)

    for k in xrange(0, len(data), batch_size):
        sample = idx[slice(k, k+batch_size)]
        yield data[sample], targets[sample]


def neural_network():
    net = nn.layers.InputLayer(shape=(None, 4))
    net = nn.layers.DenseLayer(
        incoming=net, num_units=50, nonlinearity=nn.nonlinearities.rectify)
    net = nn.layers.DenseLayer(
        incoming=net, num_units=3, nonlinearity=nn.nonlinearities.softmax)

    return net


def network_functions(net):
    X = T.fmatrix(); Y = T.ivector();
    output = nn.layers.get_output(layer_or_layers=net, inputs=X)
    loss = nn.objectives.categorical_crossentropy(output, Y).mean()
    accuracy = T.mean(T.eq(T.argmax(output, axis=1), Y))
    params = nn.layers.get_all_params(net, trainable=True)
    updates = nn.updates.sgd(loss_or_grads=loss, params=params,
        learning_rate=.01)
    train = theano.function([X,Y], [loss, accuracy], updates=updates)
    test = theano.function([X,Y], [loss, accuracy])

    return train, test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_iris_data()
    net = neural_network()
    train, test = network_functions(net)
    headers = ['Epochs', 'TL', 'TA', 'VL', 'VA', 'Time']
    s = "{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}".format(*headers)
    report(s, './output/iris_details.txt')
    TL, TA, VL, VA = [], [], [], []

    for e in range(300):
        start = time.time()

        tl, ta, t_batches = 0., 0., 0
        for data, targets in generate_minibatches(X_train, Y_train):
            l, a = train(data, targets)
            tl += l
            ta += a
            t_batches += 1

        tl /= t_batches
        ta /= t_batches
        TL.append(tl)
        TA.append(ta)

        vl, va, v_batches = 0., 0., 0
        for data, targets in generate_minibatches(X_test, Y_test):
            l, a = test(data, targets)
            vl += l
            va += a
            v_batches += 1

        vl /= v_batches
        va /= v_batches
        VL.append(vl)
        VA.append(va)

        end = time.time() - start
        s = "{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}".format(e+1,tl,ta,vl,va,end)
        report(s, './output/iris_details.txt')

    plt.switch_backend('Agg')
    plt.plot([(100 - va*100) for va in VA], c='r', lw=1.5)
    plt.plot([(100 - ta*100) for ta in TA], c='g', lw=1.5)
    plt.legend(['validation', 'training'], fontsize=20)
    # plt.ylim(-1, 10)
    plt.xlabel('Epoch', size=20)
    plt.ylabel('Error (%)', size=20)
    plt.tight_layout()
    plt.savefig('./output/iris_error.jpg', format='jpg')
    plt.savefig('./output/iris_error.eps', format='eps')