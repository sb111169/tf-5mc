import pdb

from keras.optimizer_v2.adam import Adam
from tensorboard.compat.proto.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession

from transformer import build_model

import numpy as np

import keras

from sklearn.model_selection import KFold, StratifiedKFold

from keras.models import Model

# from keras.optimizers import Adam
# from keras.optimizers.optimizer_v2.adam import Adam
from keras.optimizers import adam_v2
from keras.preprocessing.text import one_hot
from keras.layers import Dense, Dropout, MaxPooling1D, Attention, Flatten, Input, merge, MultiHeadAttention, \
    LayerNormalization, Embedding, Reshape
from keras.layers import Conv1D, LSTM, concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import os
#
#
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(
#     config=config)  # 默认开启 Tensorflow 的 session 之后，就会占用几乎所有的显存，这样的话速度会比较快。使用allow_growth，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片（当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存）

splitcharBy = 3  # 拆分比
overlap_interval = 1  # 重叠间隔

window = 2  # 窗口
# try:
#     roc_auc_score(y_true,y_pred)
# except ValueError:
#     pass



def cal_base(y_true, y_pred):
    y_pred_positive = np.round(np.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = np.sum(y_positive * y_pred_positive)
    TN = np.sum(y_negative * y_pred_negative)

    FP = np.sum(y_negative * y_pred_positive)
    FN = np.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN


def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP)
    return SP

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]


m5c_list_train = list(open(r'neg_train.txt', 'r'))
# print(m5c_list_train[1500])
len_seq = 41
num_in = len(m5c_list_train)
label = []
feature = []
#
random.shuffle(m5c_list_train)
for i in range(num_in):
    # seq = str(m5c_list_train[i].seq)
    seq = str(m5c_list_train[i][0:41])
    # print(seq)
    seq = seq.replace('-', 'X')  # turn rna seq to dna seq if have
    # print(m5c_list_train[i].id)
    TempArray = [seq[j:j + splitcharBy] for j in range(0, len(seq) - (len(seq) % splitcharBy), overlap_interval)]
    feature.append(TempArray)
    if m5c_list_train[i][-2] == '1':
        label.append(1)
    else:
        label.append(0)

m5c_list_test = list(open(r'neg_test.txt', 'r'))
# len_seq=len(m5c_list_train[0].seq)
len_seq = 41
num_in = len(m5c_list_test)
Y_test = []  # X_test,Y_test
X_test = []

random.shuffle(m5c_list_test)
for i in range(num_in):
    # seq = str(m5c_list_train[i].seq)
    seq = str(m5c_list_test[i][0:41])
    seq = seq.replace('-', 'X')  # turn rna seq to dna seq if have
    # print(m5c_list_train[i].id)
    TempArray = [seq[j:j + splitcharBy] for j in range(0, len(seq) - (len(seq) % splitcharBy), overlap_interval)]
    X_test.append(TempArray)
    if m5c_list_test[i][-2] == '1':
        Y_test.append(1)
    else:
        Y_test.append(0)


def load_embedding_vectors(filename):
    # load embedding_vectors from the word2vec
    # initial matrix with random uniform
    embedding_vectors = dict()
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedding_vectors[word] = vector

    f.close()
    return embedding_vectors


wordembedding = load_embedding_vectors("vectors.txt")
feature = np.array(feature)
label = np.array(label)




###########################
def generate_arrays_from_feature(feature, label, batch_size):
    while 1:
        train_data = []
        train_label = []
        cnt = 0
        for i in range(len(feature)):
            temp_list = []
            for j in feature[i]:
                word = j
                if word in wordembedding.keys():
                    temp_list.append(wordembedding[word])
                else:
                    word = "<unk>"
                    temp_list.append(wordembedding[word])
            train_data.append(temp_list)  #
            train_label.append(label[i])
            # print(train_data)
            # print(train_label)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(train_data), np.array(train_label))

                train_data = []
                train_label = []


#################5折
testAcc1 = 0
testTime1 = 0
seed = 100
kfold = 5
kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
label_predict = np.zeros(label.shape)
# Loop through the indices the split() method returns
foldi = 0
label_pre = []
cvscores = []
cvroc_auc_score = []
cvmatthews_corrcoef = []
cvaccuracy_score = []
cvprecision_score = []
cvrecall_score = []
cvspe = []
cvsen = []
cvaccuracy_score1 = []
cvprecision_score1 = []
cvmatthews_corrcoef1 = []
# cvroc_auc_score1 = []
cvspe1 = []
cvsen1 = []
roc_auc_scores_max = 0
thres = 0.9

batch_size = 16
epochs = 50


transformer_encoder = build_model(
    (19, 96),
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25
)


def ANN(optimizer='adam', neurons=64, kernel_size=5, batch_size=64, epochs=60, activation='relu', patience=50, drop=0.2,
        loss='categorical_crossentropy', hidden_size=128):
    vocab_size = 65
    embed_dim = 64

    filters = 64
    num_heads = 2
    ff_dim = 16

    inp1 = Input(shape=(39, 300))
    ######### model1
    cnn1 = Conv1D(32, 3, padding='same', strides=1, activation='relu')(inp1)
    cnn3 = MaxPooling1D(pool_size=2)(cnn1)
    drop1 = Dropout(0.2)(cnn3)

    cnn11 = Conv1D(32, 3, padding='same', strides=1, dilation_rate=2, activation='relu')(inp1)
    cnn13 = MaxPooling1D(pool_size=2)(cnn11)
    drop11 = Dropout(0.2)(cnn13)

    cnn21 = Conv1D(32, 3, padding='same', strides=1, dilation_rate=3, activation='relu')(inp1)
    cnn23 = MaxPooling1D(pool_size=2)(cnn21)
    drop21 = Dropout(0.2)(cnn23)
    cnn = concatenate([drop1, drop11, drop21], axis=-1)

    x = transformer_encoder(cnn)
    # x = cnn
    flat = Flatten()(x)
    x1 = Dense(128, activation='relu')(flat)
    x2 = Dropout(0.5)(x1)
    x3 = Dense(64, activation='relu')(x2)
    x4 = Dropout(0.5)(x3)
    x5 = Dense(32, activation='relu')(x4)
    x6 = Dropout(0.5)(x5)
    final_output = Dense(1, activation='sigmoid')(x6)
    model = Model(inputs=inp1, outputs=final_output)  #

    model.summary()

    return model


model = ANN()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=3e-4),
              metrics=METRICS)

earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
callbacks = [earlystopping]
features = generate_arrays_from_feature(feature, label, batch_size)
# import pdb
# pdb.set_trace()
validation_features = generate_arrays_from_feature(X_test, Y_test, batch_size)

# model.fit_generator(features,  # fit_generator
#                     epochs=epochs,
#                     steps_per_epoch=len(feature) // batch_size,
#                     verbose=2,
#                     callbacks=callbacks,
#                     validation_data=generate_arrays_from_feature(X_test, Y_test, batch_size),
#                     validation_steps=len(X_test) // batch_size)

model.fit(
    features,  # fit_generator
    epochs=epochs,
    steps_per_epoch=len(feature) // batch_size,
    verbose=2,
    callbacks=callbacks,
    validation_data=validation_features,
    validation_steps=len(X_test) // batch_size
)
# ####save
model.save('m1A-RGloVe-model1.h5')
