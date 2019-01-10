# Correlation Matrix Visualization 相关性矩阵可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('../input/jinnan_round1_testA_20181227.csv', encoding='gb18030')
cate_columns = [f for f in train.columns if f != '样本id']
corr = train[cate_columns].corr()
corr.fillna(0)
corr.to_csv("../input/corr.csv", index=False, header=None)
%matplotlib inline
sns.set(rc={'figure.figsize':(50,50)})
g=sns.heatmap(corr, ax=fig,annot=True, fmt=".2f")
