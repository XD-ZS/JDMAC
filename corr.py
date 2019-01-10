import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../data/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('../data/jinnan_round1_testA_20181227.csv', encoding='gb18030')
cate_columns = [f for f in train.columns if f != '样本id']
corr = train[cate_columns].corr()
corr.fillna(0)
corr.to_csv("../data/corr.csv", index=False, header=None)
f,fig=plt.subplots(figsize=(50,50),nrows=1)
# sns.set(rc={'figure.figsize':(50,50)})
fig.tick_params(axis='x',labelsize=20) # x轴
fig.tick_params(axis='y',labelsize=20) # y轴
sns.set(font_scale=1.5)
g=sns.heatmap(corr, ax=fig,annot=True, fmt=".2f")
f.savefig('../data/corr.jpg', dpi=100, bbox_inches='tight')
