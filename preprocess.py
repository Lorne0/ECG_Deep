import numpy as np
import sys, os
import wfdb
import pywt
import pickle as pk
from collections import Counter

if len(sys.argv)!=(5+1):
    print("Usage: python preprocess.py <NLRAV/NSVFQ> <denoise/no> <normalize/no> <augment/no> <random_seed>")
    exit(-1)

mode = sys.argv[1] # NLRAV / NSVFQ
if_denoise = sys.argv[2]
if_normal = sys.argv[3]
if_augment = sys.argv[4]
r_seed = int(sys.argv[5])
print(sys.argv)

def denoise(sig, sigma, wn='bior1.3'):
    threshold = sigma * np.sqrt(2*np.log2(len(sig)))
    c = pywt.wavedec(sig, wn)
    thresh = lambda x: pywt.threshold(x,threshold,'soft')
    nc = list(map(thresh, c))
    return pywt.waverec(nc, wn)

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
        if d!='114': # since 114 is reversed
            sig = np.array(r.p_signal[:,0])
        else:
            sig = np.array(r.p_signal[:,1])
        if if_denoise=='denoise':
            sig = denoise(sig, 0.005, 'bior1.3')
        if if_normal=='normalize':
            sig = (sig-min(sig)) / (max(sig)-min(sig))
        sig_len = len(sig)
        sym = ann.symbol
        pos = ann.sample
        beat_len = len(sym)
        for i in range(beat_len):
            if sym[i] in labels and pos[i]-wid>=0 and pos[i]+wid+1<=sig_len:
                a = sig[pos[i]-wid:pos[i]+wid+1]
                if len(a) != 2*wid+1:
                    print("Length error")
                    continue
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
        if d!='114':
            sig = np.array(r.p_signal[:,0])
        else:
            sig = np.array(r.p_signal[:,1])
        if if_denoise=='denoise':
            sig = denoise(sig, 0.005, 'bior1.3')
        if if_normal=='normal':
            sig = (sig-min(sig)) / (max(sig)-min(sig))
        sig_len = len(sig)
        sym = ann.symbol
        pos = ann.sample
        beat_len = len(sym)
        for i in range(beat_len):
            if sym[i] in sub_labels and pos[i]-wid>=0 and pos[i]+wid+1<=sig_len:
                a = sig[pos[i]-wid:pos[i]+wid+1]
                if len(a) != 2*wid+1:
                    print("Length error")
                    continue
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

train_len = int(data_len*0.6) # 60%
valid_len = int(data_len*0.2) # 20%
test_len = data_len-train_len-valid_len # 20%

X_train = X[idx][:train_len]
X_valid = X[idx][train_len:train_len+valid_len]
X_test = X[idx][train_len+valid_len:]
Y_train = Y[idx][:train_len]
Y_valid = Y[idx][train_len:train_len+valid_len]
Y_test = Y[idx][train_len+valid_len:]

if if_augment=='augment':
    X = np.zeros((train_len*9, 2*wid+1))
    Y = np.zeros((train_len*9))
    cnt = 0
    for i in range(train_len):
        label = Y_train[i]
        if label==0: # only -N-LRAV or -N-SVFQ need to do augment
            c = X_train[i].copy()
            X[cnt] = c
            Y[cnt] = label
            cnt += 1
            continue
            
        c = X_train[i].copy() # center
        t = c+(np.max(c)-np.min(c))*0.1 # top
        b = c-(np.max(c)-np.min(c))*0.1 # bottom
        head = c[0]
        tail = c[-1]
        lc = np.concatenate( (c[int(wid/5):], [tail]*int(wid/5)) ) # left center
        rc = np.concatenate( ([head]*int(wid/5), c[:-int(wid/5)]) ) # right center
        lt = lc+(np.max(lc)-np.min(lc))*0.1 # left top
        lb = lc-(np.max(lc)-np.min(lc))*0.1 # left bottom
        rt = rc+(np.max(rc)-np.min(rc))*0.1 # right top
        rb = rc-(np.max(rc)-np.min(rc))*0.1 # right bottom

        X[cnt] = c
        Y[cnt] = label
        cnt += 1
        X[cnt] = t
        Y[cnt] = label
        cnt += 1
        X[cnt] = b
        Y[cnt] = label
        cnt += 1
        X[cnt] = lc
        Y[cnt] = label
        cnt += 1
        X[cnt] = lt
        Y[cnt] = label
        cnt += 1
        X[cnt] = lb
        Y[cnt] = label
        cnt += 1
        X[cnt] = rc
        Y[cnt] = label
        cnt += 1
        X[cnt] = rt
        Y[cnt] = label
        cnt += 1
        X[cnt] = rb
        Y[cnt] = label
        cnt += 1
    X_train = X[:cnt]
    Y_train = Y[:cnt]
    train_len = len(X_train)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(Counter(Y_train))
print(Counter(Y_valid))
print(Counter(Y_test))


X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

fn = "data_"+mode+".pk"
with open(fn, "wb") as fw:
    pk.dump(X_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_test, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_test, fw, protocol=pk.HIGHEST_PROTOCOL)


      



