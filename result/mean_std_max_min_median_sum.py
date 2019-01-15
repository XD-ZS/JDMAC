# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 100)

train = pd.read_csv('../data/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('../data/jinnan_round1_testA_20181227.csv', encoding='gb18030')

# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
good_cols = list(train.columns)
# 删除某一类别占比超过90%的列
# for col in train.columns:
#     rate = train[col].value_counts(normalize=True, dropna=False).values[0]
#     if rate > 0.9:
#         good_cols.remove(col)
#         print(col, rate)
good_cols.remove('B2')
# 删除异常值
train = train[train['收率'] > 0.87]

train = train[good_cols]
good_cols.remove('收率')
test = test[good_cols]
# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.fillna(-1)

def proportion_A234(t):
    try:
#         print(type(t['A1']),type(t['A2']),type(t['A3']),type(t['A4']))
        if t['A2']==-1.0:
            return float(t['A3'])/float(t['A4'])
        else:
            return float(t['A2'])/float(t['A4'])
    except:
        return 0
def proportion_A1234(t):
    try:
        if t['A2']==-1.0:
            return (float(t['A1'])+float(t['A3']))/float(t['A4'])
        else:
            return (float(t['A1'])+float(t['A2']))/float(t['A4'])
    except:
        return 0

data['A2_3_4']=data.apply(proportion_A234, axis=1)
data['A1_2_3_4']=data.apply(proportion_A1234, axis=1)
# 去掉'A1','A2','A3','A4'
# data.drop(['A1','A2','A3','A4'], axis=1, inplace=True)
def timeTranSecond(t):
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0

    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600

    return tm


for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f, '应该在前面被删除了！')


def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1

    return tm


for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)


def getInterval(train, before, after):
    t1 = train[before]
    t2 = train[after]
    if t2 < t1:
        return t2 - t1 + 24
    elif t2 >= t1:
        return t2 - t1


a = ['A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
for inx, val in enumerate(a):
    if (val == 'B7'):
        break
    data[str(inx)] = data.apply(getInterval, axis=1, args=(val, a[int(inx) + 1]))

data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))

categorical_columns = [f for f in data.columns if f not in ['样本id']]
numerical_columns = [f for f in data.columns if f not in categorical_columns]

data['b14/a1_a3_a4_a19_b1_b12'] = data['B14'] / (
            data['A1'] + data['A3'] + data['A4'] + data['A19'] + data['B1'] + data['B12'])

numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')
del data['A1']
del data['A3']
del data['A4']
for f in ['A1','A3','A4']:
    if f in categorical_columns:
        categorical_columns.remove(f)
# categorical_columns.remove('A1')
# categorical_columns.remove('A3')
# categorical_columns.remove('A4')
# label encoder
for f in categorical_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
train = data[:train.shape[0]]
test = data[train.shape[0]:]
print(train.shape)
print(test.shape)
# train['target'] = list(target)
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']

total_columns={}

for feat in  ['B14']:
    for f1 in categorical_columns:
        cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name={}
                col_name['mean']   = feat+'_to_' + f1 + "_" + f2 + '_mean'
                col_name['std']    = feat+'_to_' + f1 + "_" + f2 + '_std'
                col_name['max']    = feat + '_to_' + f1 + "_" + f2 + '_max'
                col_name['min']    = feat + '_to_' + f1 + "_" + f2 + '_min'
                col_name['median'] = feat + '_to_' + f1 + "_" + f2 + '_median'
                col_name['sum']    = feat + '_to_' + f1 + "_" + f2 + '_sum'

                total_columns.setdefault('mean', []).append(col_name['mean'])
                total_columns.setdefault('std', []).append(col_name['std'])
                total_columns.setdefault('max', []).append(col_name['max'])
                total_columns.setdefault('min', []).append(col_name['min'])
                total_columns.setdefault('median', []).append(col_name['median'])
                total_columns.setdefault('sum', []).append(col_name['sum'])

                order_label={}
                order_label['mean']=[]
                order_label['std'] = []
                order_label['max'] = []
                order_label['min'] = []
                order_label['median'] = []
                order_label['sum'] = []

                order_label['mean']   = train.groupby([f1])[f2].mean()
                order_label['std']    = train.groupby([f1])[f2].std()
                order_label['max']    = train.groupby([f1])[f2].max()
                order_label['min']    = train.groupby([f1])[f2].min()
                order_label['median'] = train.groupby([f1])[f2].median()
                order_label['sum']    = train.groupby([f1])[f2].sum()

                train[col_name['mean']] = train[feat].map(order_label['mean'])
                train[col_name['std']] = train[feat].map(order_label['std'])
                train[col_name['max']] = train[feat].map(order_label['max'])
                train[col_name['min']] = train[feat].map(order_label['min'])
                train[col_name['median']] = train[feat].map(order_label['median'])
                train[col_name['sum']] = train[feat].map(order_label['sum'])

                miss_rate={}
                miss_rate['mean']   = train[col_name['mean']].isnull().sum() * 100 / train[col_name['mean']].shape[0]
                miss_rate['std']    = train[col_name['std']].isnull().sum() * 100 / train[col_name['std']].shape[0]
                miss_rate['max']    = train[col_name['max']].isnull().sum() * 100 / train[col_name['max']].shape[0]
                miss_rate['min']    = train[col_name['min']].isnull().sum() * 100 / train[col_name['min']].shape[0]
                miss_rate['median'] = train[col_name['median']].isnull().sum() * 100 / train[col_name['median']].shape[0]
                miss_rate['sum']    = train[col_name['sum']].isnull().sum() * 100 / train[col_name['sum']].shape[0]

                for m_rate in miss_rate:
                    # print(m_rate)
                    if miss_rate[m_rate] > 0:
                        train = train.drop([feat+'_to_' + f1 + "_" + f2 + "_"+m_rate], axis=1)
                        total_columns.setdefault(m_rate, []).remove(col_name[m_rate])
                    else:
                        test[feat+'_to_' + f1 + "_" + f2 + '_'+m_rate] = test[feat].map(order_label[m_rate])

train.drop(li + ['target'], axis=1, inplace=True)
print(train.shape)
print(test.shape)
t_columns=total_columns['mean']+total_columns['std']+total_columns['max']+total_columns['min']+total_columns['median']+total_columns['sum']
X_train = train[t_columns + numerical_columns].values
X_test = test[t_columns + numerical_columns].values
# one hot
enc = OneHotEncoder()
for f in categorical_columns:
    enc.fit(data[f].values.reshape(-1, 1))
    X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
    X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
y_train = target.values
def modeling_cross_validation(params, X, y, nr_folds=5):
    oof_preds = np.zeros(X.shape[0])
    folds = KFold(n_splits=nr_folds, shuffle=False, random_state=4096)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])

        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=100)
        oof_preds[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)

    score = mean_squared_error(oof_preds, target)
    return score / 2
def featureSelect(init_cols):
    params = {'num_leaves': 120,
              'min_data_in_leaf': 30,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.05,
              "min_child_samples": 30,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              "metric": 'mse',
              "lambda_l1": 0.02,
              "verbosity": -1}
    best_cols = init_cols.copy()
    best_score = modeling_cross_validation(params, train[init_cols].values, target.values, nr_folds=5)
    print("初始CV score: {:<8.8f}".format(best_score))
    for f in init_cols:

        best_cols.remove(f)
        score = modeling_cross_validation(params, train[best_cols].values, target.values, nr_folds=5)
        diff = best_score - score
        print('-' * 10)
        if diff > 0.0000002:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 有效果,删除！！".format(f, score, best_score))
            best_score = score
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f, score, best_score))
            best_cols.append(f)
    print('-' * 10)
    print("优化后CV score: {:<8.8f}".format(best_score))

    return best_cols

best_features = featureSelect(train.columns.tolist())
print(best_features)
# best_features=['样本id', 'A2', 'A5', 'A6', 'A7', 'A9', 'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17', 'A19', 'A20', 'A21', 'A22', 'A26', 'A27', 'A28', 'B1', 'B4', 'B5', 'B6', 'B8', 'B10', 'B11', 'B12', 'B14', 'A2_3_4', 'A1_2_3_4', '0', '1', '2', '3', '4', '5', '6', 'b14/a1_a3_a4_a19_b1_b12', 'B14_to_A5_intTarget_0.0_mean', 'B14_to_A5_intTarget_1.0_mean', 'B14_to_A5_intTarget_2.0_mean', 'B14_to_A5_intTarget_3.0_mean', 'B14_to_A5_intTarget_4.0_mean', 'B14_to_A6_intTarget_0.0_mean', 'B14_to_A6_intTarget_1.0_mean', 'B14_to_A6_intTarget_2.0_mean', 'B14_to_A6_intTarget_3.0_mean', 'B14_to_A6_intTarget_4.0_mean', 'B14_to_A7_intTarget_0.0_mean', 'B14_to_A7_intTarget_1.0_mean', 'B14_to_A7_intTarget_2.0_mean', 'B14_to_A7_intTarget_3.0_mean', 'B14_to_A7_intTarget_4.0_mean', 'B14_to_A9_intTarget_0.0_mean', 'B14_to_A9_intTarget_1.0_mean', 'B14_to_A9_intTarget_2.0_mean', 'B14_to_A9_intTarget_3.0_mean', 'B14_to_A9_intTarget_4.0_mean', 'B14_to_A11_intTarget_0.0_mean', 'B14_to_A11_intTarget_1.0_mean', 'B14_to_A11_intTarget_2.0_mean', 'B14_to_A11_intTarget_3.0_mean', 'B14_to_A11_intTarget_4.0_mean', 'B14_to_A14_intTarget_0.0_mean', 'B14_to_A14_intTarget_1.0_mean', 'B14_to_A14_intTarget_2.0_mean', 'B14_to_A14_intTarget_3.0_mean', 'B14_to_A14_intTarget_4.0_mean', 'B14_to_A16_intTarget_0.0_mean', 'B14_to_A16_intTarget_1.0_mean', 'B14_to_A16_intTarget_2.0_mean', 'B14_to_A16_intTarget_3.0_mean', 'B14_to_A16_intTarget_4.0_mean', 'B14_to_A24_intTarget_0.0_mean', 'B14_to_A24_intTarget_1.0_mean', 'B14_to_A24_intTarget_2.0_mean', 'B14_to_A24_intTarget_3.0_mean', 'B14_to_A24_intTarget_4.0_mean', 'B14_to_A26_intTarget_0.0_mean', 'B14_to_A26_intTarget_1.0_mean', 'B14_to_A26_intTarget_2.0_mean', 'B14_to_A26_intTarget_3.0_mean', 'B14_to_A26_intTarget_4.0_mean', 'B14_to_B1_intTarget_0.0_mean', 'B14_to_B1_intTarget_1.0_mean', 'B14_to_B1_intTarget_2.0_mean', 'B14_to_B1_intTarget_3.0_mean', 'B14_to_B1_intTarget_4.0_mean', 'B14_to_B5_intTarget_0.0_mean', 'B14_to_B5_intTarget_1.0_mean', 'B14_to_B5_intTarget_2.0_mean', 'B14_to_B5_intTarget_3.0_mean', 'B14_to_B5_intTarget_4.0_mean', 'B14_to_B6_intTarget_0.0_mean', 'B14_to_B6_intTarget_1.0_mean', 'B14_to_B6_intTarget_2.0_mean', 'B14_to_B6_intTarget_3.0_mean', 'B14_to_B6_intTarget_4.0_mean', 'B14_to_B7_intTarget_0.0_mean', 'B14_to_B7_intTarget_1.0_mean', 'B14_to_B7_intTarget_2.0_mean', 'B14_to_B7_intTarget_3.0_mean', 'B14_to_B7_intTarget_4.0_mean', 'B14_to_B8_intTarget_0.0_mean', 'B14_to_B8_intTarget_1.0_mean', 'B14_to_B8_intTarget_2.0_mean', 'B14_to_B8_intTarget_3.0_mean', 'B14_to_B8_intTarget_4.0_mean', 'B14_to_B14_intTarget_0.0_mean', 'B14_to_B14_intTarget_1.0_mean', 'B14_to_B14_intTarget_2.0_mean', 'B14_to_B14_intTarget_3.0_mean', 'B14_to_B14_intTarget_4.0_mean', 'B14_to_3_intTarget_0.0_mean', 'B14_to_3_intTarget_1.0_mean', 'B14_to_3_intTarget_2.0_mean', 'B14_to_3_intTarget_3.0_mean', 'B14_to_3_intTarget_4.0_mean', 'B14_to_4_intTarget_0.0_mean', 'B14_to_4_intTarget_1.0_mean', 'B14_to_4_intTarget_2.0_mean', 'B14_to_4_intTarget_3.0_mean', 'B14_to_4_intTarget_4.0_mean', 'B14_to_5_intTarget_0.0_mean', 'B14_to_5_intTarget_1.0_mean', 'B14_to_5_intTarget_2.0_mean', 'B14_to_5_intTarget_3.0_mean', 'B14_to_5_intTarget_4.0_mean', 'B14_to_6_intTarget_0.0_mean', 'B14_to_6_intTarget_1.0_mean', 'B14_to_6_intTarget_2.0_mean', 'B14_to_6_intTarget_3.0_mean', 'B14_to_6_intTarget_4.0_mean']
X_train = train[best_features].values
X_test = test[best_features].values
# one hot
enc = OneHotEncoder()
for f in categorical_columns:
    enc.fit(data[f].values.reshape(-1, 1))
    X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
    X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
y_train = target.values


param = {'num_leaves': 120,
         'min_data_in_leaf': 30,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))
##### xgb
xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))
# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

m1 = mean_squared_error(target.values, oof_stack)
print(m1)
sub_df = pd.read_csv('../data/jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = predictions
sub_df[1] = sub_df[1].apply(lambda x: round(x, 3))
sub_df.to_csv("../data/test/" + str(m1) + ".csv", index=False, header=None)
