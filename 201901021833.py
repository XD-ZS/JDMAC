# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

train = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding='gb18030')

# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
    
# 删除缺失率超过90%的列
good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        good_cols.remove(col)

# 删除异常值
train = train[train['收率']>0.87]
        
train = train[good_cols]
good_cols.remove('收率')
test  = test[good_cols]
# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)
# 日期中有些输入错误和遗漏
def t2s(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600
        elif t=='1900/1/1 2:30':
            return 2*3600+30*60
        elif t==-1:
            return -1
        else:
            return 0
    
    try:
        tm = int(t)*3600+int(m)*60+int(s)
    except:
        return 30*60
    
    return tm
for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
    data[f] = data[f].apply(t2s)

def getDuration(se):
    try:
        sh,sm,eh,em=re.split("[:,-]",se)
    except:
        if se=='14::30-15:30':
            return 3600
        elif se=='13；00-14:00':
            return 3600
        elif se=='21:00-22；00':
            return 3600
        elif se=='22"00-0:00':
            return 7200
        elif se=='2:00-3;00':
            return 3600
        elif se=='1:30-3;00':
            return 5400
        elif se=='15:00-1600':
            return 3600
        elif se==-1:
            return -1
        else:
            return 30*60
        
    try:
        tm = int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600
    except:
        if se=='19:-20:05':
            return 3600
        else:
            return 30*60
    
    return tm
for f in ['A20','A28','B4','B9','B10','B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)

cate_columns = [f for f in data.columns if f != '样本id']

#label encoder
for f in cate_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
train = data[:train.shape[0]]
test  = data[train.shape[0]:]

# one-hot
# X_train = pd.DataFrame()
# X_test = pd.DataFrame()
# enc = OneHotEncoder()
# for f in cate_columns:
#     enc.fit(data[f].values.reshape(-1, 1))
#     X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
#     X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')

# y_train = target.values

param = {'num_leaves': 30,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}

# 五折交叉验证
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, 
                    trn_data, 
                    num_round, 
                    valid_sets = [trn_data, val_data], 
                    verbose_eval = 200, 
                    early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    
    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)))

import time
time_name = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))

# 提交结果
sub_df = pd.read_csv('./data/jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = predictions
sub_df[1] = sub_df[1].apply(lambda x:(round(x, 3)))
sub_df.to_csv("./data/" + time_name + ".csv", index=False, header=None)