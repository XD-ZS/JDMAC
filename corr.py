# Correlation Matrix Visualization 相关性矩阵可视化
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
train = pd.read_csv('../input/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('../input/jinnan_round1_testA_20181227.csv', encoding='gb18030')
cate_columns = [f for f in train.columns if f != '样本id']
#print(cate_columns)
#for columns in cate_columns:
#    print(columns,train[columns].unique())
corr = train[cate_columns].corr()
corr.fillna(0)
corr.to_csv("../input/corr.csv", index=False, header=None)
%matplotlib inline
sns.set(rc={'figure.figsize':(50,50)})
g=sns.heatmap(corr, ax=fig,annot=True, fmt=".2f")
