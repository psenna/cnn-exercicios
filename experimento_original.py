from __future__ import print_function

# Experimento original do artigo

# Tensorflow
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

import numpy as np

from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Activation, Flatten, concatenate, UpSampling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import *
from keras.layers import Lambda

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

import src.DataViconLoad as DataViconLoad    # Import the data
from src.Split import Split
from src.temporal_sample import TemporalSample
from src.split_body_parts import SplitBodyParts
from src.temporal_pyramid import TempPyramid

import datetime
now = datetime.datetime.now


timesteps = 240  # Numero de amostras por exercício
nr = 90   # Número de repetições
n_dim = 117  # Tamanho do vetor de característica (39 relações entre juntas x 3 ângulos)


Correct_data, Correct_label, Incorrect_data, Incorrect_label = DataViconLoad.load_data('./')

# Print the size of the data
print(Correct_data.shape, 'correct sequences')
print(Correct_label.shape, 'correct labels')
print(Incorrect_data.shape, 'incorrect sequences')
print(Incorrect_label.shape, 'incorrect labels')

train_x, train_y, valid_x, valid_y = Split.original(Correct_data, Correct_label, Incorrect_data, Incorrect_label, nr)

train_x, valid_x, train_x_2, valid_x_2, train_x_4, valid_x_4, train_x_8, valid_x_8 = TemporalSample.original(train_x, valid_x)

trainx = SplitBodyParts.reorder_data(train_x)
validx = SplitBodyParts.reorder_data(valid_x)
trainx_2 = SplitBodyParts.reorder_data(train_x_2)
validx_2 = SplitBodyParts.reorder_data(valid_x_2)
trainx_4 = SplitBodyParts.reorder_data(train_x_4)
validx_4 = SplitBodyParts.reorder_data(valid_x_4)
trainx_8 = SplitBodyParts.reorder_data(train_x_8)
validx_8 = SplitBodyParts.reorder_data(valid_x_8)

n_dim = 90 # dimension after segmenting the data into body parts
n_dim1 = 12 # trunk dimension
n_dim2 = 18 # arms dimension
n_dim3 = 21 # legs dimension

# Build the model ...

#### Full scale sequences
seq_input = Input(shape = (timesteps, n_dim), name = 'full_scale')

seq_input_trunk = Lambda(lambda x: x[:, :, 0:12])(seq_input)
seq_input_left_arm = Lambda(lambda x: x[:, :, 12:30])(seq_input)
seq_input_right_arm = Lambda(lambda x: x[:, :, 30:48])(seq_input)
seq_input_left_leg = Lambda(lambda x: x[:, :, 48:69])(seq_input)
seq_input_right_leg = Lambda(lambda x: x[:, :, 69:90])(seq_input)

#### Half scale sequences
seq_input_2 = Input(shape=(int(timesteps/2), n_dim), name='half_scale')

seq_input_trunk_2 = Lambda(lambda x: x[:, :, 0:12])(seq_input_2)
seq_input_left_arm_2 = Lambda(lambda x: x[:, :, 12:30])(seq_input_2)
seq_input_right_arm_2 = Lambda(lambda x: x[:, :, 30:48])(seq_input_2)
seq_input_left_leg_2 = Lambda(lambda x: x[:, :, 48:69])(seq_input_2)
seq_input_right_leg_2 = Lambda(lambda x: x[:, :, 69:90])(seq_input_2)

#### Quarter scale sequences
seq_input_4 = Input(shape=(int(timesteps/4), n_dim), name='quarter_scale')

seq_input_trunk_4 = Lambda(lambda x: x[:, :, 0:12])(seq_input_4)
seq_input_left_arm_4 = Lambda(lambda x: x[:, :, 12:30])(seq_input_4)
seq_input_right_arm_4 = Lambda(lambda x: x[:, :, 30:48])(seq_input_4)
seq_input_left_leg_4 = Lambda(lambda x: x[:, :, 48:69])(seq_input_4)
seq_input_right_leg_4 = Lambda(lambda x: x[:, :, 69:90])(seq_input_4)

#### Eighth scale sequences
seq_input_8 = Input(shape=(int(timesteps/8), n_dim), name='eighth_scale')

seq_input_trunk_8 = Lambda(lambda x: x[:, :, 0:12])(seq_input_8)
seq_input_left_arm_8 = Lambda(lambda x: x[:, :, 12:30])(seq_input_8)
seq_input_right_arm_8 = Lambda(lambda x: x[:, :, 30:48])(seq_input_8)
seq_input_left_leg_8 = Lambda(lambda x: x[:, :, 48:69])(seq_input_8)
seq_input_right_leg_8 = Lambda(lambda x: x[:, :, 69:90])(seq_input_8)

concat_trunk = TempPyramid(seq_input_trunk, seq_input_trunk_2, seq_input_trunk_4, seq_input_trunk_8, timesteps, n_dim1)
concat_left_arm = TempPyramid(seq_input_left_arm, seq_input_left_arm_2, seq_input_left_arm_4, seq_input_left_arm_8, timesteps, n_dim2)
concat_right_arm = TempPyramid(seq_input_right_arm, seq_input_right_arm_2, seq_input_right_arm_4, seq_input_right_arm_8, timesteps, n_dim2)
concat_left_leg = TempPyramid(seq_input_left_leg, seq_input_left_leg_2, seq_input_left_leg_4, seq_input_left_leg_8, timesteps, n_dim3)
concat_right_leg = TempPyramid(seq_input_right_leg, seq_input_right_leg_2, seq_input_right_leg_4, seq_input_right_leg_8, timesteps, n_dim3)

concat = concatenate([concat_trunk, concat_left_arm, concat_right_arm, concat_left_leg, concat_right_leg], axis=-1)

rec = LSTM(80, return_sequences=True)(concat)
rec1 = LSTM(40, return_sequences=True)(rec)
rec1 = LSTM(40, return_sequences=True)(rec1)
rec2 = LSTM(80)(rec1)

out = Dense(1, activation='sigmoid')(rec2)

model = Model(inputs=[seq_input, seq_input_2, seq_input_4, seq_input_8], outputs=out)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001))

t = now()

early_stopping = EarlyStopping(monitor='val_loss', patience=25)

history = model.fit([trainx, trainx_2, trainx_4, trainx_8], train_y, batch_size=3, epochs=500, verbose=0,
                    validation_data=([validx, validx_2, validx_4, validx_8], valid_y), callbacks=[early_stopping])

print('Training time: %s' % (now() - t))

# Plot the results
plt.figure(1)
plt.plot(history.history['loss'], 'b', label = 'Training Loss')
plt.title('Training Loss')
plt.plot(history.history['val_loss'], 'r', label = 'Validation Loss')
plt.legend()
plt.tight_layout()
# plt.show()

# Print the minimum loss
print("Training loss", np.min(history.history['loss']))
print("Validation loss",np.min(history.history['val_loss']))

# Plot the prediction of the model for the training and validation sets
pred_train = model.predict([trainx, trainx_2, trainx_4, trainx_8])

pred_test = model.predict([validx, validx_2, validx_4, validx_8])

plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(pred_train,'s', color='red', label='Prediction', linestyle='None', alpha = 0.5, markersize=6)
plt.plot(train_y,'o', color='green',label='Quality Score', alpha = 0.4, markersize=6)
plt.ylim([-0.1,1.1])
plt.title('Training Set',fontsize=18)
plt.xlabel('Sequence Number',fontsize=16)
plt.ylabel('Quality Scale',fontsize=16)
plt.legend(loc=3, prop={'size':14}) # loc:position
plt.subplot(2,1,2)
plt.plot(pred_test,'s', color='red', label='Prediction', linestyle='None', alpha = 0.5, markersize=6)
plt.plot(valid_y,'o', color='green',label='Quality Score', alpha = 0.4, markersize=6)
plt.title('Testing Set',fontsize=18)
plt.ylim([-0.1,1.1])
plt.xlabel('Sequence Number',fontsize=16)
plt.ylabel('Quality Scale',fontsize=16)
plt.legend(loc=3, prop={'size':14}) # loc:position
plt.tight_layout()
plt.savefig('./Results/Original.png', dpi=300)
# plt.show()

# Calculate the cumulative deviation and rms deviation for the validation set
test_dev = abs(np.squeeze(pred_test)-valid_y)
# Cumulative deviation
mean_abs_dev = np.mean(test_dev)
# RMS deviation
rms_dev = sqrt(mean_squared_error(pred_test, valid_y))
print('Mean absolute deviation:', mean_abs_dev)
print('RMS deviation:', rms_dev)