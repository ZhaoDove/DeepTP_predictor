#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DeepTP_predictor 
@File    ：test.py
@Author  ：Dove
@Date    ：2023/1/5 22:20 
'''
import os
import pickle
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.preprocessing import sequence

from sklearn.preprocessing import LabelEncoder, RobustScaler
rbs = RobustScaler()


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"step_dim": self.step_dim, "W_regularizer": self.W_regularizer, "b_regularizer": self.b_regularizer,
                  "W_constraint": self.W_constraint, "b_constraint": self.b_constraint, "bias": self.bias}

        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Encoding1
def trans(str12):
    if type(str12) != str:
        print("start")
    # print(f'{type(str12)}')
    a = []
    dic = {'A': 1, 'B': 0, 'U': 0, 'J': 0, 'Z': 0, 'O': 0, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
           'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
           'X': 0}
    for i in range(len(str12)):
        a.append(dic.get(str12[i]))
    return a


def Encoding(test):
    test['encode'] = test['Sequence'].apply(lambda x: trans(x))
    return test


# Encoding2
def trans_6(str1):
    a = []
    dic = {'A': 6, 'C': 1, 'D': 2, 'E': 2, 'F': 1, 'G': 4, 'H': 3, 'I': 1, 'K': 3, 'L': 1, 'M': 1, 'N': 5, 'P': 4,
           'Q': 5, 'R': 3, 'S': 5, 'T': 6, 'V': 1, 'W': 1, 'Y': 1, 'X': 0, 'B': 0, 'U': 0, 'J': 0, 'Z': 0, 'O': 0, }
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a


def Encoding_6(test):
    test['encode1'] = test['Sequence'].apply(lambda x: trans_6(x))
    return test


def zero_or_one(x):
    return 1 if x > 0.5 else 0


model = tf.keras.models.load_model('../prediction_model/model_DeepTP.h5', custom_objects={'Attention': Attention})

def test():
    fea_csv_path = '../example/example.csv'
    if not os.path.exists(fea_csv_path):
        f = zipfile.ZipFile("../example/example.zip", 'r')
        for file in f.namelist():
            f.extract(file, "../example/")
        f.close()
    test_data = pd.read_csv(fea_csv_path)

    fea_csv_path1 = '../dataset/biological features for training.csv'
    if not os.path.exists(fea_csv_path1):
        f = zipfile.ZipFile("../dataset/biological features for training.zip", 'r')
        for file in f.namelist():
            f.extract(file, "../dataset/")
        f.close()
    train_data = pd.read_csv(fea_csv_path1, index_col=0)

    train_features = train_data.drop(['UniProt_id', 'Type', 'Sequence'], axis=1).select_dtypes(exclude=['object'])
    test_features = test_data.drop(['UniProt_id', 'Type', 'Sequence'], axis=1).select_dtypes(exclude=['object'])
    train_features = train_features.values
    test_features = test_features.values
    train_features_rbs = rbs.fit_transform(train_features)
    test_features_rbs = rbs.transform(test_features)
    model_file_name = "../feature_selection/selector_RFECV_205.pickle"
    with open(model_file_name, 'rb') as f:
        get_model = pickle.load(f)
    support = get_model.get_support(True)
    x_bio_test = test_features_rbs[:, support]

    test_y = test_data["Type"]
    test_y = test_y.values

    max_length = 1500
    test1 = Encoding(test_data)
    test_encode = test1['encode']
    test_encode = sequence.pad_sequences(test_encode, maxlen=max_length)
    test_encode = np.reshape(test_encode, (test_encode.shape[0], 1500, 1))

    test2 = Encoding_6(test_data)
    test_encode1 = test2['encode1']
    test_encode1 = sequence.pad_sequences(test_encode1, maxlen=max_length)
    test_encode1 = np.reshape(test_encode1, (test_encode1.shape[0], 1500, 1))

    y_pred = model.predict([test_encode, test_encode1, x_bio_test])
    result1 = list(map(zero_or_one, y_pred))

    result1 = pd.DataFrame(result1)
    y_pred1 = pd.DataFrame(y_pred)
    Result = pd.concat([test_data[['UniProt_id']], result1, y_pred1], axis=1)
    Result.columns = ['UniProt_id', 'predict_result', 'score']
    Result.to_csv(r'../predict_result.csv')

if __name__ == '__main__':
    test()