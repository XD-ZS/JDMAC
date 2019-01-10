import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('../data/jinnan_round1_testA_20181227.csv', encoding='gb18030')
f,ax1=plt.subplots(figsize=(100,100),nrows=1)
# fig.tick_params(axis='x',labelsize=1000) # x轴
# fig.tick_params(axis='y',labelsize=1000) # y轴,这里没用
train.hist(ax=ax1,xlabelsize =50,ylabelsize =50)
f.savefig("../data/train_hist.png")
f,ax2=plt.subplots(figsize=(100,100),nrows=1)
test.hist(ax=ax2,xlabelsize =50,ylabelsize =50)
f.savefig("../data/test_hist.png")
