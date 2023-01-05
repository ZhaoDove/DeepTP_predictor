# -*- coding: UTF-8 -*-
'''
@Project ：DeepTP_test_20221120 
@File    ：test.py
@Author  ：Dove
@Date    ：2022/11/20 20:24 
'''

# 基本库
import os
import time
import joblib
import pickle

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from tensorflow.python.keras.models import Model

label_encoder = LabelEncoder()

from sklearn import metrics
from sklearn.feature_selection import RFECV, RFE

import pandas as pd
import numpy as np

np.random.seed(1337)  # for reproducibility

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras import initializers, regularizers, constraints

from sklearn.preprocessing import RobustScaler

rbs = RobustScaler()


# self-attention
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


# Encoding1 氨基酸组成编码
def trans(str1):
    a = []
    dic = {'A': 1, 'B': 0, 'U': 0, 'J': 0, 'Z': 0, 'O': 0, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
           'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
           'X': 0}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a


def Encoding(test):
    # 全序列
    test['encode'] = test['Sequence'].apply(lambda x: trans(x))
    return test


# Encoding2 氨基酸理化性质编码
def trans_6(str1):
    a = []
    dic = {'A': 6, 'C': 1, 'D': 2, 'E': 2, 'F': 1, 'G': 4, 'H': 3, 'I': 1, 'K': 3, 'L': 1, 'M': 1, 'N': 5, 'P': 4,
           'Q': 5, 'R': 3, 'S': 5, 'T': 6, 'V': 1, 'W': 1, 'Y': 1, 'X': 0, 'B': 0, 'U': 0, 'J': 0, 'Z': 0, 'O': 0, }
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a


def Encoding_6(test):
    # 全序列
    test['encode1'] = test['Sequence'].apply(lambda x: trans_6(x))
    return test


# 目前用的模型--特征直接concat
def build_model():
    bio_input = Input(shape=(205,), name='bio_input')
    print('bio_input', bio_input.shape)

    sequence_input = Input(shape=(1500,), name='sequence_input')
    print('sequence_input', sequence_input.shape)

    six_input = Input(shape=(1500,), name='six_input')
    print('six_input', six_input.shape)

    embedding = Embedding(21, 128)(sequence_input)
    embedding = Dropout(0.5)(embedding)
    print('embedding_seq', embedding.shape)
    conv1 = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(embedding)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling1D()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling1D()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv2)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling1D()(conv3)
    sequence_out = Bidirectional(LSTM(64, return_sequences=True))(conv3)
    sequence_out1 = Bidirectional(LSTM(64))(conv3)
    sequence_out = Attention(187)(sequence_out)

    sequence_out2 = Concatenate()([sequence_out, sequence_out1])

    print('sequence_out', sequence_out.shape)
    print('sequence_out1', sequence_out1.shape)
    print('sequence_out2', sequence_out1.shape)

    embedding_six = Embedding(21, 128)(six_input)
    embedding_six = Dropout(0.5)(embedding_six)
    conv1_six = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(embedding_six)
    conv1_six = Activation('relu')(conv1_six)
    conv1_six = MaxPooling1D()(conv1_six)
    conv2_six = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv1_six)
    conv2_six = Activation('relu')(conv2_six)
    conv2_six = MaxPooling1D()(conv2_six)
    conv3_six = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv2_six)
    conv3_six = Activation('relu')(conv3_six)
    conv3_six = MaxPooling1D()(conv3_six)
    six_out = Bidirectional(LSTM(64, return_sequences=True))(conv3_six)
    six_out1 = Bidirectional(LSTM(64))(conv3_six)
    six_out = Attention(187)(six_out)
    six_out2 = Concatenate()([six_out, six_out1])

    print('six_out', six_out.shape)
    print('six_out1', six_out1.shape)
    print('six_out2', six_out2.shape)

    out = Concatenate()([sequence_out2, six_out2])
    concat_bio = Concatenate()([out, bio_input])
    feature_output = LayerNormalization()(concat_bio)

    feature_output = Dense(256, activation='relu')(feature_output)
    feature_output = Dropout(0.5)(feature_output)
    feature_output = Dense(128, activation='relu')(feature_output)
    feature_output = Dropout(0.5)(feature_output)
    feature_output = Dense(64, activation='relu')(feature_output)

    feature_output = Dropout(0.2)(feature_output)
    y = Dense(1, activation='sigmoid')(feature_output)

    model = Model(inputs=[sequence_input, six_input, bio_input], outputs=[y])
    adam = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


model = build_model()
model.summary()



