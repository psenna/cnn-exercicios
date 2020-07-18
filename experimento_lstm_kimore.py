from __future__ import print_function

# Experimento original do artigo

# Tensorflow
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

from sklearn.model_selection import train_test_split

import numpy as np

from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Activation, Flatten, concatenate, UpSampling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import *
from keras.layers import Lambda

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

import src.DataKimoreLoad as DataLoad    # Import the data
from src.Split import Split
from src.temporal_sample import TemporalSample
from src.split_body_parts import SplitBodyParts
from src.temporal_pyramid import TempPyramid

import datetime
now = datetime.datetime.now

X_train, y_train = DataLoad.load_data()

train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.3)
test_x, valid_x, test_y, valid_y = train_test_split(valid_x, valid_y, test_size=0.5)
print('Training data', train_x.shape)
print('Validation data', valid_x.shape)
print('Test data', test_x.shape)

# Reduce the data length by a factor of 2, 4, and 8
# The reduced sequences will be used as inputs to the temporal pyramid subnetwork
train_x_2 = train_x[:,::2,:]
valid_x_2 = valid_x[:,::2,:]
test_x_2 = test_x[:,::2,:]
train_x_4 = train_x[:,::4,:]
valid_x_4 = valid_x[:,::4,:]
test_x_4 = test_x[:,::4,:]
train_x_8 = train_x[:,::8,:]
valid_x_8 = valid_x[:,::8,:]
test_x_8 = test_x[:,::8,:]

# Code to re-order the 88 dimensional skeleton data from Kinect into trunk, left arm, right arm, left leg and right leg
def reorder_data(x):
    X_trunk = x[:,:,0:16]
    X_left_arm = x[:,:,16:32]
    X_right_arm = x[:,:,32:48]
    X_left_leg = x[:,:,48:64]
    X_right_leg = x[:,:,64:80]
    x_segmented = np.concatenate((X_trunk, X_right_arm, X_left_arm, X_right_leg, X_left_leg),axis = -1)
    return x_segmented

# Reorder the data dimensions to correspond to the five body parts
trainx = reorder_data(train_x)
validx = reorder_data(valid_x)
testx = reorder_data(test_x)
trainx_2 =  reorder_data(train_x_2)
validx_2 =  reorder_data(valid_x_2)
testx_2 =  reorder_data(test_x_2)
trainx_4 =  reorder_data(train_x_4)
validx_4 =  reorder_data(valid_x_4)
testx_4 =  reorder_data(test_x_4)

n_dim = 80 # dimension after reordering the data into body parts
n_dim1 = 16 # dimension of indiviudal body parts
seq_len = 100 # Tamanho da sequencia
lstm_mult = 0.5 # Multiplica quantidade de neurônios nas camadas lstm


# Build the model ...

#### Full scale sequences
seq_input = Input(shape = (seq_len,n_dim), name = 'full_scale')

seq_input_trunk = Lambda(lambda x: x[:, :, 0:16])(seq_input)
seq_input_left_arm = Lambda(lambda x: x[:, :, 16:32])(seq_input)
seq_input_right_arm = Lambda(lambda x: x[:, :, 32:48])(seq_input)
seq_input_left_leg = Lambda(lambda x: x[:, :, 48:64])(seq_input)
seq_input_right_leg = Lambda(lambda x: x[:, :, 64:80])(seq_input)

#### Half scale sequences
seq_input_2 = Input(shape=(int(seq_len/2), n_dim), name='half_scale')

seq_input_trunk_2 = Lambda(lambda x: x[:, :, 0:16])(seq_input_2)
seq_input_left_arm_2 = Lambda(lambda x: x[:, :, 16:32])(seq_input_2)
seq_input_right_arm_2 = Lambda(lambda x: x[:, :, 32:48])(seq_input_2)
seq_input_left_leg_2 = Lambda(lambda x: x[:, :, 48:64])(seq_input_2)
seq_input_right_leg_2 = Lambda(lambda x: x[:, :, 64:80])(seq_input_2)

#### Quarter scale sequences
seq_input_4 = Input(shape=(int(seq_len/4), n_dim), name='quarter_scale')

seq_input_trunk_4 = Lambda(lambda x: x[:, :, 0:16])(seq_input_4)
seq_input_left_arm_4 = Lambda(lambda x: x[:, :, 16:32])(seq_input_4)
seq_input_right_arm_4 = Lambda(lambda x: x[:, :, 32:48])(seq_input_4)
seq_input_left_leg_4 = Lambda(lambda x: x[:, :, 48:64])(seq_input_4)
seq_input_right_leg_4 = Lambda(lambda x: x[:, :, 64:80])(seq_input_4)


concat_trunk = TempPyramid(seq_input_trunk, seq_input_trunk_2, seq_input_trunk_4, seq_len, n_dim1)
concat_left_arm = TempPyramid(seq_input_left_arm, seq_input_left_arm_2, seq_input_left_arm_4, seq_len, n_dim1)
concat_right_arm = TempPyramid(seq_input_right_arm, seq_input_right_arm_2, seq_input_right_arm_4, seq_len, n_dim1)
concat_left_leg = TempPyramid(seq_input_left_leg, seq_input_left_leg_2, seq_input_left_leg_4, seq_len, n_dim1)
concat_right_leg = TempPyramid(seq_input_right_leg, seq_input_right_leg_2, seq_input_right_leg_4, seq_len, n_dim1)

concat = concatenate([concat_trunk, concat_left_arm, concat_right_arm, concat_left_leg, concat_right_leg], axis=-1)

rec = LSTM(int(80*lstm_mult), return_sequences=True)(concat)
rec1 = LSTM(int(40*lstm_mult), return_sequences=True)(rec)
rec1 = LSTM(int(40*lstm_mult), return_sequences=True)(rec1)
rec2 = LSTM(int(80*lstm_mult))(rec1)

out = Dense(1, activation = 'sigmoid')(rec2)

model = Model(inputs=[seq_input, seq_input_2, seq_input_4], outputs=out)

model.compile(loss='binary_crossentropy', optimizer= Adam(lr=0.0001))

t = now()

early_stopping = EarlyStopping(monitor='val_loss', patience=25)

history = model.fit([trainx, trainx_2, trainx_4], train_y, batch_size=10, epochs=500, verbose=0,
                    validation_data=([validx, validx_2, validx_4], valid_y), callbacks=[early_stopping])

print('Training time: %s' % (now() - t))

# Plot the results
plt.figure(1)
plt.plot(history.history['loss'], 'b', label = 'Loss Treinamento')
plt.title('Loss Treinamento')
plt.plot(history.history['val_loss'], 'r', label = 'Loss Validação')
plt.legend()
plt.tight_layout()
plt.show()

# Print the minimum loss
print("Training loss", np.min(history.history['loss']))
print("Validation loss",np.min(history.history['val_loss']))


# Plot the prediction of the model for the training and validation sets
pred_train = model.predict([trainx, trainx_2, trainx_4])

pred_valid = model.predict([validx, validx_2, validx_4])

pred_test = model.predict([testx, testx_2, testx_4])

pred_grafico = concatenate([pred_valid, pred_test])
label_grafico = concatenate([valid_y, test_y])

plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(pred_train,'s', color='red', label='Previsões', linestyle='None', alpha = 0.5, markersize=6)
plt.plot(train_y,'o', color='green',label='Avaliações', alpha = 0.4, markersize=6)
plt.ylim([-0.1,1.1])
plt.title('Dados de Treino',fontsize=18)
plt.xlabel('Sequencias',fontsize=16)
plt.ylabel('Avaliação',fontsize=16)
plt.legend(loc=3, prop={'size':14}) # loc:position
plt.subplot(2,1,2)
plt.plot(pred_grafico,'s', color='red', label='Previsões', linestyle='None', alpha = 0.5, markersize=6)
plt.plot(label_grafico,'o', color='green',label='Avaliações', alpha = 0.4, markersize=6)
plt.title('Dados de Validação e Teste',fontsize=18)
plt.ylim([-0.1,1.1])
plt.xlabel('Sequencias (31 primeiras validação)',fontsize=16)
plt.ylabel('Avaliação',fontsize=16)
plt.legend(loc=3, prop={'size':14}) # loc:position
plt.tight_layout()
plt.show()

# Calculate the cumulative deviation and rms deviation for the validation set
test_dev = abs(np.squeeze(pred_valid)-valid_y)
# Cumulative deviation
mean_abs_dev = np.mean(test_dev)
# RMS deviation
rms_dev = sqrt(mean_squared_error(pred_valid, valid_y))
print('Validacao Mean absolute deviation:', mean_abs_dev)
print('Validacao RMS deviation:', rms_dev)

# Calculate the cumulative deviation and rms deviation for the test set
test_dev = abs(np.squeeze(pred_test)-test_y)
# Cumulative deviation
mean_abs_dev = np.mean(test_dev)
# RMS deviation
rms_dev = sqrt(mean_squared_error(pred_test, test_y))
print('Test Mean absolute deviation:', mean_abs_dev)
print('Test RMS deviation:', rms_dev)