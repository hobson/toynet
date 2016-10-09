import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import ParametricSoftExponential, ParametricSoftplus
from keras.regularizers import l1, activity_l1, l1l2

M = 30000
N = 1
nb_epoch = 100
num_experiments = 5


results = []
for i in range(num_experiments):
    X = pd.DataFrame(pd.np.random.randn(M, 2))
    y = pd.DataFrame((X.T.values[:N] * X.T.values[N:])).sum()
    row = []
    for act in [ParametricSoftExponential(0.0), ParametricSoftplus(), Activation('relu'), Activation('linear')]:
        model = Sequential()
        model.add(Dense(2 * N, input_dim=2 * N, W_regularizer=l1l2(0.005)))
        model.add(act)
        model.add(Dense(1 * N, input_dim=2 * N, W_regularizer=l1l2(0.005)))
        model.add(act)
        model.add(Dense(1, input_dim=1 * N, W_regularizer=None))  # l1l2(0.003)))
        model.add(Activation('linear'))

        # model.add(Dense(2, input_dim=2, W_regularizer=l1l2(0.001)))
        # # model.add(ParametricSoftplus(.2))
        # model.add(ParametricSoftExponential(.9, output_dim=1))
        # model.add(Dense(1, input_dim=2, W_regularizer=l1l2(0.001)))
        # # model.add(ParametricSoftplus(.2))
        # model.add(ParametricSoftExponential(-.9))
        # model.add(Dense(1, input_dim=2, W_regularizer=l1l2(0.001)))
        # model.add(Activation('relu'))

        # solution varies even when training data is unchanged
        # this will converge to < 0.01 loss about 60% of the time, NaNs about 20% of the time, and RMSE loss 1.-3. for the remainder
        model.compile(loss='mean_squared_error', optimizer='rmsprop')  # SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
        model.fit(X.values, y.values, batch_size=32, validation_split=0, nb_epoch=nb_epoch)
        row += [model.evaluate(X.values, y.values)]
    print(row)
    results += [row]
