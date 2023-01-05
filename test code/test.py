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