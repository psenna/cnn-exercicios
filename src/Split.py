# Métodos de divisão dos dados para treino/teste
import numpy as np
import random

class Split():
    @classmethod
    def original(cls, Correct_data, Correct_label, Incorrect_data, Incorrect_label, nr):
        # Método original do artigo
        # Split the data into training and validation sets
        # Training set: 70%
        # Validation set: 30%

        # Sample random indices
        trainidx1 = random.sample(range(0, Correct_data.shape[0]), int(nr * 0.7))
        trainidx2 = random.sample(range(0, Incorrect_data.shape[0]), int(nr * 0.7))
        valididx1 = np.setdiff1d(np.arange(0, nr, 1), trainidx1)
        valididx2 = np.setdiff1d(np.arange(0, nr, 1), trainidx2)

        # Training set: data and labels
        train_x = np.concatenate((Correct_data[trainidx1, :, :], Incorrect_data[trainidx2, :, :]))
        print(train_x.shape, 'training data')
        train_y = np.concatenate((np.squeeze(Correct_label[trainidx1]), np.squeeze(Incorrect_label[trainidx2])))
        print(train_y.shape, 'training labels')

        # Validation set: data and labels
        valid_x = np.concatenate((Correct_data[valididx1, :, :], Incorrect_data[valididx2, :, :]))
        print(valid_x.shape, 'validation data')
        valid_y = np.concatenate((np.squeeze(Correct_label[valididx1]), np.squeeze(Incorrect_label[valididx2])))
        print(valid_y.shape, 'validation labels')
        return train_x, train_y, valid_x, valid_y

    @classmethod
    def treino_validacao_teste(cls, Correct_data, Correct_label, Incorrect_data, Incorrect_label, nr):
        # Divide em treino, validação e teste
        # Training set: 70%
        # Validation set: 15%
        # Test set: 15%

        # Sample random indices
        trainidx1 = random.sample(range(0, Correct_data.shape[0]), int(nr * 0.7))
        trainidx2 = random.sample(range(0, Incorrect_data.shape[0]), int(nr * 0.7))
        valididx1 = np.setdiff1d(np.arange(0, nr, 1), trainidx1)
        valididx2 = np.setdiff1d(np.arange(0, nr, 1), trainidx2)

        # Training set: data and labels
        train_x = np.concatenate((Correct_data[trainidx1, :, :], Incorrect_data[trainidx2, :, :]))
        print(train_x.shape, 'training data')
        train_y = np.concatenate((np.squeeze(Correct_label[trainidx1]), np.squeeze(Incorrect_label[trainidx2])))
        print(train_y.shape, 'training labels')

        tmp_correct = Correct_data[valididx1, :, :]
        tmp_incorrect = Incorrect_data[valididx2, :, :]
        tmp_correct_labels = np.squeeze(Correct_label[valididx1])
        tmp_incorrect_labels = np.squeeze(Incorrect_label[valididx2])

        testx1 = random.sample(range(0, tmp_correct.shape[0]), int(nr * 0.15))
        testx2 = random.sample(range(0, tmp_incorrect.shape[0]), int(nr * 0.15))
        valididx1 = np.setdiff1d(np.arange(0, tmp_correct.shape[0], 1), testx1)
        valididx2 = np.setdiff1d(np.arange(0, tmp_correct.shape[0], 1), testx2)

        # Test set: data and labels
        test_x = np.concatenate((tmp_correct[testx1, :, :], tmp_incorrect[testx2, :, :]))
        print(test_x.shape, 'test data')
        test_y = np.concatenate((np.squeeze(tmp_correct_labels[testx1]), np.squeeze(tmp_incorrect_labels[testx2])))
        print(test_y.shape, 'test labels')

        # Validation set: data and labels
        valid_x = np.concatenate((tmp_correct[valididx1, :, :], tmp_incorrect[valididx2, :, :]))
        print(valid_x.shape, 'validation data')
        valid_y = np.concatenate((np.squeeze(tmp_correct_labels[valididx1]), np.squeeze(tmp_incorrect_labels[valididx2])))
        print(valid_y.shape, 'validation labels')
        return train_x, train_y, valid_x, valid_y, test_x, test_y
