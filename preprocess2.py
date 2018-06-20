import numpy as np
import sys, os
import wfdb
import pywt
import pickle as pk
from collections import Counter

if len(sys.argv)!=(2+1):
    print("Usage: python preprocess2.py <NLRAV/NSVFQ> <random_seed>")
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

X_train = X[idx][:train_len]
X_valid = X[idx][train_len:train_len+valid_len]
X_test = X[idx][train_len+valid_len:]
Y_train = Y[idx][:train_len]
Y_valid = Y[idx][train_len:train_len+valid_len]
Y_test = Y[idx][train_len+valid_len:]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(Counter(Y_train))
print(Counter(Y_valid))
print(Counter(Y_test))

fn = "data2_"+mode+".pk"
with open(fn, "wb") as fw:
    pk.dump(X_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_test, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_test, fw, protocol=pk.HIGHEST_PROTOCOL)


      



