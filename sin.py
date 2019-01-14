#p226
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.layers.recurrent import SimpleRNN, GRU
from keras.initializers import TruncatedNormal
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# sin波のTを求めるネットワーク
# input: length = 100の信号
# output: T

def sin_predict(maxlen=100, shift=0, T=100):
    x = np.arange(0, maxlen)
    data = np.zeros((1, maxlen, 2))
    data[0, :, 0] = np.sin(2.0 * np.pi * (x + shift) / T)
    data[0, :, 1] = shift
    return data


def sin(x, shift=0, T=100):
    return np.sin(2.0 * np.pi * (x + shift) / T) # sampling定理は無視

def row_data(N, maxlen):
    data = np.zeros((N, maxlen, 2))
    target = np.zeros((N, 1))
    for t in np.arange(0, 10000):
        x = np.arange(0, maxlen)
        freq = t % (maxlen + 1) + 1 # 1 - 100
        shift = np.random.randint(maxlen)
        data[t, :, 0] = sin(x, shift=shift, T=freq)
        data[t, :, 1] = shift
        target[t] = freq
    return data, target

def execute_train(model, X_train, Y_train, batch_size, epochs, X_validation, Y_validation, loss_graph_filename='fighre.png'):
    early_stopping = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validation, Y_validation), callbacks=[early_stopping])
    plt.plot(range(len(hist.history['val_loss'])), hist.history['val_loss'], label='acc', color='black')
    plt.xlabel('epochs')
    plt.savefig(loss_graph_filename)

# gru
def trainer1(X, Y, N, maxlen):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1)
    print(X_train.shape, X_validation.shape, Y_train.shape, Y_validation.shape)

    n_in = 2
    n_hidden = 100
    n_out = 1

    epochs = 200
    batch_size = 100

    model = Sequential([
        GRU(n_hidden, input_shape=(maxlen, n_in)),
        Dense(n_out),
        Activation('linear')
    ])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
    execute_train(model, X_train, Y_train, batch_size, epochs, X_validation, Y_validation, 'loss_gru.png')

    print ('predict')
    for i in range(10, 20):
        expected = i
        predicted =  model.predict(sin_predict(shift=0, T=expected))
        print('expected:', expected, 'predict:', predicted)


# dnn 2 layers
def trainer2(X, Y, N, maxlen):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1)
    print(X_train.shape, X_validation.shape, Y_train.shape, Y_validation.shape)
    X_train = X_train[:, :, 0].reshape(int(N * 0.9), maxlen)
    X_validation = X_validation[:, :, 0].reshape(N - int(N * 0.9), maxlen)

    n_in = 100
    n_hidden = 100
    n_out = 1
    model = Sequential([
        Dense(n_hidden, input_shape=(maxlen,), kernel_initializer=TruncatedNormal(stddev=0.01)),
        Activation('relu'),
        Dense(n_hidden, kernel_initializer=TruncatedNormal(stddev=0.01)),
        Activation('relu'),
        Dense(n_out, kernel_initializer=TruncatedNormal(stddev=0.01)),
        Activation('linear')
    ])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
    epochs = 200
    batch_size = 100
    execute_train(model, X_train, Y_train, batch_size, epochs, X_validation, Y_validation, 'loss_dnn.png')

    print ('predict')
    for i in range(10, 20):
        expected = i
        predicted =  model.predict(sin_predict(shift=0, T=expected)[0, :, 0].reshape(1, maxlen))
        print('expected:', expected, 'predict:', predicted)

# simple rnn
def trainer3(X, Y, N, maxlen):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1)
    print(X_train.shape, X_validation.shape, Y_train.shape, Y_validation.shape)

    n_in = 2
    n_hidden = 100
    n_out = 1

    epochs = 200
    batch_size = 100

    model = Sequential([
        SimpleRNN(n_hidden, input_shape=(maxlen, n_in)),
        Dense(n_out),
        Activation('linear')
    ])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
    execute_train(model, X_train, Y_train, batch_size, epochs, X_validation, Y_validation, 'loss_rnn.png')

    print ('predict')
    for i in range(10, 20):
        expected = i
        predicted =  model.predict(sin_predict(shift=0, T=expected))
        print('expected:', expected, 'predict:', predicted)

    N = 10000
    maxlen = 100
    X, Y = row_data(N, maxlen)

if __name__ == '__main__':

    N = 10000
    maxlen = 100
    X, Y = row_data(N, maxlen)

    trainer1(X, Y, N, maxlen)
    trainer2(X, Y, N, maxlen)
    trainer3(X, Y, N, maxlen)

