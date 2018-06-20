import numpy as np
import os, sys

fn = sys.argv[1] # result_#

SE = 0.0
ACC = 0.0
AUC = 0.0
SP = 0.0
vSE = 0.0
vACC = 0.0
cnt = 0
with open("./result/"+fn) as fp:
    for f in fp:
        s = f.strip("\n").split('|')
        SE += float(s[0].split(':')[1])
        ACC += float(s[1].split(':')[1])
        AUC += float(s[2].split(':')[1])
        SP += float(s[3].split(':')[1])
        vSE += float(s[4].split(':')[1])
        vACC += float(s[5].split(':')[1])
        cnt += 1
SE /= cnt
ACC /= cnt
AUC /= cnt
SP /= cnt
vSE /= cnt
vACC /= cnt


print(fn.split('_')[1]+": SE: %.4f | ACC: %.4f | AUC: %.4f | SP: %.4f | valid SE: %.4f | valid ACC: %.4f" %(SE, ACC, AUC, SP, vSE, vACC))

            

