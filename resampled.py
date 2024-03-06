import numpy as np
import pandas as pd
def resampled(X_train,y_train):
    l = []
    l1 = []
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    # 分开少数类，默认标签为1的为少数类
    ss = pd.to_numeric(y_train.iloc[:, 0])
    for i, element in enumerate(ss):
        if element == 1:
            l.append(i)
    X_train1 = X_train.iloc[l, :]
    y_train1 = y_train.iloc[l, :]
    X_train1 = X_train1.reset_index(drop=True)
    y_train1 = y_train1.reset_index(drop=True)
    for j in l:
        X_train.drop(index=j, inplace=True)
        y_train.drop(index=j, inplace=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    # 多数类和少数类合并为新样本，前面为多数类后面为少数类，方便判断反近邻
    X_resampled = pd.concat([X_train, X_train1])
    y_resampled = pd.concat([y_train, y_train1])
    X_resampled = X_resampled.reset_index(drop=True)
    y_resampled = y_resampled.reset_index(drop=True)
    return(X_resampled,y_resampled)