#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DeepTP_predictor 
@File    ：test.py
@Author  ：Dove
@Date    ：2023/1/5 22:20 
'''
# 评价指标
import pandas as pd
from tensorflow import metrics
import tensorflow as tf
from tensorflow.keras import initializers, constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *


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


def tolerance_metrics(y_true, y_pre):
    """
    计算 tolerance 需要的12个指标
    :param y_true: 标签的正确值
    :param y_pre: 标签的预测值
    :return : (六个指标) -> (DateFrame)
    """
    label = pd.DataFrame({'true': y_true, 'pre': y_pre})

    # 计算每一类的 TP、FN、FP、TN
    unique_state = label.true.unique()  # 寻找到唯一的标签
    targets = {}  # 保存结果
    state_map = {1: 'p', 0: 'n', '0': 'p', '0': 'n'}
    tp = fp = tn = fn = 0
    for i, (t, p) in label.iterrows():
        # i, t, p -> 索引，真实值，预测值
        if t == 0 and p == 0:
            tn += 1
        if t == 0 and p == 1:
            fp += 1
        if t == 1 and p == 1:
            tp += 1
        if t == 1 and p == 0:
            fn += 1

    allp = tp + fn
    alln = fp + tn

    # 7个计算指标
    N = tp + tn + fp + fn
    print("tp", tp, "tn", tn, "fp", fp, "fn", fn)
    # ppv
    ppv = tp / (tp + fp)
    # npv
    npv = tn / (tn + fn)
    # sensitivity -> TPR
    sen = tp / (tp + fn)
    # spciticity -> TNR
    spe = tn / (tn + fp)
    # acc
    acc = (tp + tn) / N
    # MCC
    mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    # OPM
    opm = (ppv + npv) * (sen + spe) * (acc + (1 + mcc) / 2) / 8
    # auc
    rocauc = metrics.roc_auc_score(y_true, y_pre)
    # 构造成 pandas
    columns = ['TP', 'TN', 'FP', 'FN', 'PPV', 'NPV', 'TPR', 'TNR', 'ACC', 'MCC', 'OPM', 'AUC', 'N']
    res = pd.DataFrame(
        [
            [tp, tn, fp, fn, ppv, npv, sen, spe, acc, mcc, opm, rocauc, N]
        ],
        columns=columns,
    )
    # 其他数据（保留，暂时不会用到）

    return res.T


# 转换成标签
def zero_or_one(x):
    return 1 if x > 0.5 else 0


model = tf.keras.models.load_model('../model/model_DeepTP.h5',custom_objects={'Attention': Attention})

# Show the model architecture
model.summary()