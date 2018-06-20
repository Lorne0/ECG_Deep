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
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam

def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

if len(sys.argv)!=(2+1):
    print("Usage: python train.py <NLRAV/NSVFQ> <save_file_name>")
    exit(-1)

mode = sys.argv[1] # NLRAV / NSVFQ
save = sys.argv[2] # result_NLRAV_0

fn = "data_"+mode+".pk"
with open(fn, "rb") as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)
    X_test = pk.load(fp)
    Y_test = pk.load(fp)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
f_size = X_train.shape[1]
class_num = 5

#============================================#

lr = 0.01
batch_size=32

Y_train = keras.utils.to_categorical(Y_train, num_classes=class_num)

def make_model():
    model = Sequential()
    model.add(Conv1D(18, 7, activation='relu', input_shape=(f_size,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(18, 7, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    return model
model = make_model()

best_SE = 0
best_ACC = 0
best_model = make_model()
patience = 30
pcnt = 0

bin_label = lambda x: min(1,x)

for e in range(1, 300+1):

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=0)

    y_pred = model.predict(X_valid)
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.sum(y_pred==Y_valid)/len(Y_valid)

    y_true = list(map(bin_label, Y_valid))
    y_pred = list(map(bin_label, y_pred))
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    SE = tp/(tp+fn)
    SP = tn/(fp+tn)

    if SE+acc > best_SE+best_ACC:
        best_SE, best_ACC = SE, acc
        best_model.set_weights(model.get_weights())
        pcnt = 0
    else:
        pcnt += 1
    
    print("Epoch: %d | SE: %.4f | Best SE: %.4f | ACC: %.4f | Best ACC: %.4f | AUC: %.4f | SP: %.4f" %(e, SE, best_SE, acc, best_ACC, auc, SP))

    if pcnt==patience:
        y_pred = best_model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(y_pred==Y_test)/len(Y_test)
        y_true = list(map(bin_label, Y_test))
        y_pred = list(map(bin_label, y_pred))
        auc = roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        SE = tp/(tp+fn)
        SP = tn/(fp+tn)
        print(mode+" Test | SE: %.4f | ACC: %.4f | AUC: %.4f | SP: %.4f | valid SE: %.4f | valid ACC: %.4f" %(SE, acc, auc, SP, best_SE, best_ACC))
        with open("./result/"+save, "a") as fw:
            fw.write("SE: %.4f | ACC: %.4f | AUC: %.4f | SP: %.4f | valid SE: %.4f | valid ACC: %.4f\n" %(SE, acc, auc, SP, best_SE, best_ACC))
        break





