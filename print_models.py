import numpy as np
import pickle as pk
import os, sys
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score 
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam

def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

f_size=300
class_num=5
lr=0.005
model_type=sys.argv[1]

def make_model():
    model = Sequential()
    if model_type == '1D':
        model.add(Conv1D(18, 7, activation='relu', input_shape=(f_size,1)))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(18, 7, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(class_num, activation='softmax'))
    elif model_type == '1D-small':
        model.add(Conv1D(10, 3, activation='relu', input_shape=(f_size,1)))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(10, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(class_num, activation='softmax'))
    elif model_type == '1D-large':
        model.add(Conv1D(50, 13, activation='relu', input_shape=(f_size,1)))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(50, 13, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(class_num, activation='softmax'))
    elif model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True, dropout=0.1, input_shape=(f_size, 1)))
        model.add(LSTM(32, return_sequences=True, dropout=0.1))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(class_num, activation='softmax'))
    elif model_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.1), merge_mode='sum', input_shape=(f_size, 1)))
        model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.1), merge_mode='sum'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(class_num, activation='softmax'))
    elif model_type == 'Dense':
        model.add(Dense(256, input_dim=f_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, input_dim=f_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(class_num, activation='softmax'))
    elif model_type=='2D':
        model.add(Conv2D(64, 3, input_shape=(40,40,1)))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, 3))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='elu'))
        model.add(Dense(class_num))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    return model

model = make_model()
print(model.summary())
