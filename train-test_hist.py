import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('../data/jinnan_round1_testA_20181227.csv', encoding='gb18030')
fig,(ax1,ax2)=plt.subplots(figsize=(100,100),nrows=2)
# fig.tick_params(axis='x',labelsize=1000) # x轴
# fig.tick_params(axis='y',labelsize=1000) # y轴,这里没用
a=[]
for f in train.columns:
    if train[f].dtypes!='object':
        a.append(f)
b=train.hist(column=a,ax=ax,xlabelsize =50,ylabelsize =50)
[x.title.set_size(32) for x in b.ravel()]
fig.savefig("../data/train_hist.png")
test.hist(ax=ax2,xlabelsize =50,ylabelsize =50)
fig.savefig("../data/test_hist.png")
