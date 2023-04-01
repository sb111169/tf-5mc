# -*- coding = utf-8 -*-
# @Time : 2022/12/5 9:52
# @Author :wangzhe
# @File : getdata.py
# @Software : PyCharm
from Bio import SeqIO
import numpy as np
i = 0
j = 0
aa = {}
bb = {}
cc = {}
dd = {}
for my_aa in SeqIO.parse(r'F:\data\iPromoter-5mC-master\all_positive.fasta', 'fasta'):
    AA = str(my_aa.seq)
    if i <52000:

        aa[i] = str(AA)
    elif i<65000:
        cc[i] = str(AA)
    elif i>=12000:
        break
    i += 1
    with open('pos65000.txt', 'w') as f:
        for key, value in aa.items():
            f.write(str(value))
            f.write('\n')
    with open('pos65000_test.txt', 'w') as f:
        for key, value in cc.items():
            f.write(str(value))
            f.write('\n')

for my_bb in SeqIO.parse(r'F:\data\iPromoter-5mC-master\all_negative.fasta', 'fasta'):
    AA = str(my_bb.seq)
    if j < 640000:
        bb[j] = str(AA)
    elif j<800000:
        dd[j] = str(AA)
    elif j>=800000:
        break
    j += 1

with open('neg800000.txt', 'w') as f:
    for key, value in bb.items():
        f.write(str(value))
        f.write('\n')
with open('neg800000_test.txt', 'w') as f:
    for key, value in dd.items():
        f.write(str(value))
        f.write('\n')