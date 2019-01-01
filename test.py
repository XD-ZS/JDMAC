# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy import sparse

train = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding='gb18030')

print(train.describe())
print(test.describe())