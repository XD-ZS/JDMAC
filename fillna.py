# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
# values={'A1':300,'A3':405,'A4':700,'A6':29,'A10':100,'A12':103,'A13':0.2,
#         'A15':104,'A17':105,'A18':0.2,'A21':50.0,'A22':9.0,'A23':5.0,'A25':80,
#         'A27':73,'B1':320.0,'B6':80,'B8':45.0,'B12':1200}
# data = data.fillna(value=values)
# a=list(data.dtypes)
# for i,val in enumerate(data.columns):
#     if a[i]=='int64':
# #         print(data[val].mean())
# #         data[val]=data[val].fillna(int(data[val].mean()))
# #         data[val]=data[val].fillna(int(data[val].median()))
# #         data[val]=data[val].fillna(data[val].mode())
#     elif a[i]=='float64':
#         data[val]=data[val].fillna(data[val].mean())
#         data[val]=data[val].fillna(data[val].median())
#         data[val]=data[val].fillna(data[val].mode())
# data.fillna(method='bfill',inplace=True)
data.fillna(data.mean(), inplace=True) 
data = data.fillna(-1)
