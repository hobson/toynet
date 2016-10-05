"""Get Dense NN to learn dot product, if possible"""

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import ParametricSoftExp

np = pd.np


def multipliplay(N=1, M=1000, nb_epoch=200, activation='relu', lr=0.001, momentum=0.001, decay=0.001, nesterov=False):
    """ Learn to multiply and and inputs together """

    from keras.regularizers import l1, activity_l1

    model = Sequential()
    model.add(Dense(4 * N, input_dim=N * 2, W_regularizer=l1(0.03), activity_regularizer=activity_l1(0.03)))
    model.add(Activation(activation))
    model.add(Dense(4 * N, W_regularizer=l1(0.03), activity_regularizer=activity_l1(0.03)))
    model.add(Activation(activation))
    model.add(Dense(1, W_regularizer=l1(0.03), activity_regularizer=activity_l1(0.03)))
    model.add(Activation(activation))

    model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov),
                  loss='mse')

    X = pd.DataFrame(pd.np.random.randn(M, N * 2))
    y = (X.T.loc[:N] * X.T.loc[N:]).sum().T.values
    model.fit(X.values, y, nb_epoch=nb_epoch)

    X_test = pd.DataFrame(pd.np.random.randn(int(M * 0.1), N * 2))
    y_test = (X_test.T.loc[:N] * X_test.T.loc[N:]).sum().T.values
    print('test set loss: {}'.format(model.evaluate(X_test.values, y_test)))

    return model, X, y, X_test, y_test


def explay(N=1, M=1000, nb_epoch=200, activation=ParametricSoftExp(alpha_init=0.2), lr=0.001, momentum=0.001, decay=0.001, nesterov=False):
    """ Learn to multiply and and inputs together """

    from keras.regularizers import l1, activity_l1

    model = Sequential()
    model.add(Dense(4 * N, input_dim=N * 2, W_regularizer=l1(0.03), activity_regularizer=activity_l1(0.03)))
    model.add(Activation(activation) if isinstance(activation, str) else activation)
    model.add(Dense(4 * N, W_regularizer=l1(0.03), activity_regularizer=activity_l1(0.03)))
    model.add(Activation(activation) if isinstance(activation, str) else activation)
    model.add(Dense(1, W_regularizer=l1(0.03), activity_regularizer=activity_l1(0.03)))
    model.add(Activation(activation) if isinstance(activation, str) else activation)

    model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov),
                  loss='mse')

    X = pd.DataFrame(pd.np.random.randn(M, N * 2))
    y = (X.T.loc[:N] * X.T.loc[N:]).sum().T.values
    model.fit(X.values, y, nb_epoch=nb_epoch)

    X_test = pd.DataFrame(pd.np.random.randn(int(M * 0.1), N * 2))
    y_test = (X_test.T.loc[:N] * X_test.T.loc[N:]).sum().T.values
    print('test set loss: {}'.format(model.evaluate(X_test.values, y_test)))

    return model, X, y, X_test, y_test
