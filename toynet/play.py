import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import ParametricSoftExponential, ParametricSoftplus
from keras.regularizers import l1, activity_l1, l1l2


model = Sequential()


M = 20000
N = 1
nb_epoch = 100

model.add(Dense(1, input_dim=N * 2, W_regularizer=l1l2(0.01)))
model.add(ParametricSoftExp(1))

model.compile(loss='mean_squared_error', optimizer='rmsprop')


X = pd.DataFrame(pd.np.random.randn(M, N * 2))
y = (X.T.loc[:N] * X.T.loc[N:]).sum().T.values

model.fit(X.values, y, nb_epoch=nb_epoch)
