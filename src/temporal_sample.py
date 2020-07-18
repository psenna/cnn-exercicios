import numpy as np

class TemporalSample():
    @classmethod
    def original(cls, train_x, valid_x):
        train_x_2 = np.zeros((train_x.shape[0], int(train_x.shape[1] / 2), train_x.shape[2]))
        valid_x_2 = np.zeros(train_x_2.shape)
        train_x_4 = np.zeros((train_x.shape[0], int(train_x.shape[1] / 4), train_x.shape[2]))
        valid_x_4 = np.zeros(train_x_4.shape)
        train_x_8 = np.zeros((train_x.shape[0], int(train_x.shape[1] / 8), train_x.shape[2]))
        valid_x_8 = np.zeros(train_x_8.shape)
        train_x_2 = train_x[:, ::2, :]
        valid_x_2 = valid_x[:, ::2, :]
        train_x_4 = train_x[:, ::4, :]
        valid_x_4 = valid_x[:, ::4, :]
        train_x_8 = train_x[:, ::8, :]
        valid_x_8 = valid_x[:, ::8, :]
        return train_x, valid_x, train_x_2, valid_x_2, train_x_4, valid_x_4, train_x_8, valid_x_8

    @classmethod
    def treino_validacao_teste(cls, train_x, valid_x, test_x):
        train_x_2 = train_x[:, ::2, :]
        valid_x_2 = valid_x[:, ::2, :]
        test_x_2 = test_x[:, ::2, :]
        train_x_4 = train_x[:, ::4, :]
        valid_x_4 = valid_x[:, ::4, :]
        test_x_4 = test_x[:, ::4, :]
        train_x_8 = train_x[:, ::8, :]
        valid_x_8 = valid_x[:, ::8, :]
        test_x_8 = test_x[:, ::8, :]
        return train_x, valid_x, test_x, train_x_2, valid_x_2, test_x_2, train_x_4, valid_x_4, test_x_4, train_x_8, valid_x_8, test_x_8

