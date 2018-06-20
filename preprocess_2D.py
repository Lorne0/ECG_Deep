import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import wfdb
import pywt
import pickle as pk
from collections import Counter
from PIL import Image
from PIL import ImageOps

if len(sys.argv)!=(2+1):
    print("Usage: python preprocess_2D.py <NLRAV/NSVFQ> <random_seed>")
    exit(-1)

mode = sys.argv[1] # NLRAV / NSVFQ
r_seed = int(sys.argv[2])

data_names = ['100', '101', '102', '103', '104', '105', '106', '107', 
              '108', '109', '111', '112', '113', '114', '115', '116', 
              '117', '118', '119', '121', '122', '123', '124', '200', 
              '201', '202', '203', '205', '207', '208', '209', '210', 
              '212', '213', '214', '215', '217', '219', '220', '221', 
              '222', '223', '228', '230', '231', '232', '233', '234']

wid = 100

if mode=='NLRAV':
    labels = ['N', 'L', 'R', 'A', 'V']
    X = []
    Y = []
    for d in data_names:
        r=wfdb.rdrecord('./data/'+d)
        ann=wfdb.rdann('./data/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
        sig = np.array(r.p_signal[:,0])
        sig_len = len(sig)
        sym = ann.symbol
        pos = ann.sample
        beat_len = len(sym)
        for i in range(1,beat_len-1):
            if sym[i] in labels: 
                if (pos[i]-pos[i-1])>200 and (pos[i+1]-pos[i])>200:
                    a = sig[pos[i]-150:pos[i]+150]
                    a, cD3, cD2, cD1 = pywt.wavedec(a, 'db6', level=3)
                    X.append(a)
                    Y.append(labels.index(sym[i]))

elif mode=='NSVFQ':
    labels = ['N', 'S', 'V', 'F', 'Q']
    sub_labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
    sub = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N', 
           'A':'S', 'a':'S', 'J':'S', 'S':'S',
           'V':'V', 'E':'V',
           'F':'F',
           '/':'Q', 'f':'Q', 'Q':'Q'}
    X = []
    Y = []
    for d in data_names:
        r=wfdb.rdrecord('./data/'+d)
        ann=wfdb.rdann('./data/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
        sig = np.array(r.p_signal[:,0])
        sig_len = len(sig)
        sym = ann.symbol
        pos = ann.sample
        beat_len = len(sym)
        for i in range(1,beat_len-1):
            if sym[i] in sub_labels: 
                if (pos[i]-pos[i-1])>200 and (pos[i+1]-pos[i])>200:
                    a = sig[pos[i]-150:pos[i]+150]
                    a, cD3, cD2, cD1 = pywt.wavedec(a, 'db6', level=3)
                    X.append(a)
                    Y.append(labels.index(sub[sym[i]]))

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
print(Counter(Y))

data_len = len(X)
np.random.seed(r_seed)
idx = list(range(data_len))
np.random.shuffle(idx)

data_len = int(data_len/5)
idx = idx[:data_len]

train_len = int(data_len*0.6) #
valid_len = int(data_len*0.2)
test_len = data_len-train_len-valid_len

_X_train = X[idx][:train_len]
_X_valid = X[idx][train_len:train_len+valid_len]
_X_test = X[idx][train_len+valid_len:]
Y_train = Y[idx][:train_len]
Y_valid = Y[idx][train_len:train_len+valid_len]
Y_test = Y[idx][train_len+valid_len:]

print(_X_train.shape)
print(_X_valid.shape)
print(_X_test.shape)
print(Counter(Y_train))
print(Counter(Y_valid))
print(Counter(Y_test))

# Change 1D signal to 2D image
x = list(range(X.shape[1]))
cnt = 0

X_train = None
for i in range(len(_X_train)):
    a = _X_train[i]
    plt.clf()
    plt.figure(figsize=(0.4,0.4))
    plt.plot(x,a)
    plt.axis('off')
    fn = 'tmp_'+mode+'.png'
    plt.savefig(fn)
    plt.close()
    img = Image.open(fn).convert("L")
    img = ImageOps.invert(img)
    arr = np.asarray(img)
    arr = np.expand_dims(arr, axis=0)
    arr = np.expand_dims(arr, axis=-1)
    X_train = arr.copy() if X_train is None else np.concatenate((X_train,arr), axis=0)
    cnt += 1
    if cnt%1000==0:
        print(cnt)

X_valid = None
for i in range(len(_X_valid)):
    a = _X_valid[i]
    plt.clf()
    plt.figure(figsize=(0.4,0.4))
    plt.plot(x,a)
    plt.axis('off')
    fn = 'tmp_'+mode+'.png'
    plt.savefig(fn)
    plt.close()
    img = Image.open(fn).convert("L")
    img = ImageOps.invert(img)
    arr = np.asarray(img)
    arr = np.expand_dims(arr, axis=0)
    arr = np.expand_dims(arr, axis=-1)
    X_valid = arr.copy() if X_valid is None else np.concatenate((X_valid,arr), axis=0)
    cnt += 1
    if cnt%1000==0:
        print(cnt)

X_test = None
for i in range(len(_X_test)):
    a = _X_test[i]
    plt.clf()
    plt.figure(figsize=(0.4,0.4))
    plt.plot(x,a)
    plt.axis('off')
    fn = 'tmp_'+mode+'.png'
    plt.savefig(fn)
    plt.close()
    img = Image.open(fn).convert("L")
    img = ImageOps.invert(img)
    arr = np.asarray(img)
    arr = np.expand_dims(arr, axis=0)
    arr = np.expand_dims(arr, axis=-1)
    X_test = arr.copy() if X_test is None else np.concatenate((X_test,arr), axis=0)
    cnt += 1
    if cnt%1000==0:
        print(cnt)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

fn = "data_2D_"+mode+".pk"
with open(fn, "wb") as fw:
    pk.dump(X_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_test, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_test, fw, protocol=pk.HIGHEST_PROTOCOL)


      



