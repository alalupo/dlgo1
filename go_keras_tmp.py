import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from generate_mcts_games import data_tmp


def softmax(x):
    e_x = np.exp(x)
    print(f'e_x={e_x}')
    e_x_sum = np.sum(e_x)
    print(f'e_x_sum={e_x_sum}')
    return e_x/e_x_sum


def main():
    # data_tmp()
    x = np.array([100, 100, 100, 100, 100])
    print(f'np.sum(x)={np.sum(x)}')
    print(softmax(x))

    # np.random.seed(123)
    # X = np.load('features.npy')
    # Y = np.load('labels.npy')
    # samples = X.shape[0]
    # print(f'samples = {samples}')
    # print(f'X ndim = {X.ndim}')
    # print(f'X shape = {X.shape}')
    #
    # print(f'Y ndim = {Y.ndim}')
    # print(f'Y shape = {Y.shape}')
    # board_size = 9 * 9
    #
    # X = X.reshape(samples, board_size)
    # Y = Y.reshape(samples, board_size)

    # train_samples = int(0.9 * samples)
    # X_train, X_test = X[:train_samples], X[train_samples:]
    # Y_train, Y_test = Y[:train_samples], Y[train_samples:]
    #
    # model = Sequential()
    # model.add(Dense(1000, activation='sigmoid', input_shape=(board_size,)))
    # model.add(Dense(500, activation='sigmoid'))
    # model.add(Dense(board_size, activation='sigmoid'))
    # model.summary()
    #
    # model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    #
    # model.fit(X_train, Y_train, batch_size=64, epochs=15, verbose=1, validation_data=(X_test, Y_test))
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print(f'Test loss: {score[0]}')
    # print(f'Test accuracy: {score[1]}')


if __name__ == '__main__':
    main()
