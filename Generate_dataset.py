# 生产数据集

from Bio import SeqIO
import numpy as np


def AA_ONE_HOT(AA):
    one_hot_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    coding_arr = np.zeros((len(AA), 4), dtype=float)

    for m in range(len(AA)):
        coding_arr[m] = one_hot_dict[AA[m]]

    return coding_arr


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


i = 0
j = 0
a = np.zeros((69750, 41, 4))
a_label = np.zeros((69750, 2))
b = np.zeros((823576, 41, 4))
b_label = np.zeros((823576, 2))
aa = a.copy()
aa_label = a_label.copy()
aa_kmer= {}
bb = b.copy()
bb_label = b_label.copy()
bb_kmer= {}
for my_aa in SeqIO.parse(r'all_positive.fasta', 'fasta'):
    AA = str(my_aa.seq)

    aa_label[i] = [0,1]
    aa_kmer[i]=seq2kmer(AA, 3)
    with open('kmer_a.txt','a',encoding='utf-8') as f:
        f.write(str(aa_kmer))
        f.write('\r\n')

    i += 1

for my_bb in SeqIO.parse(r'all_negative.fasta', 'fasta'):
    AA = str(my_bb.seq)
    bb_label[j] = [1,0]
    bb_kmer[j] = seq2kmer(AA, 3)
    with open('kmer_b.txt','a',encoding='utf-8') as f:
        f.write(str(bb_kmer))
        f.write('\r\n')
    j += 1

# with open('wz.txt', 'w') as f:
#     for key, value in aa_kmer.items():
#         f.write(str(value))
#         f.write('\n')
