
import numpy as np 
import pandas as pd
import math
import os



# 在导入数据集时出现了乱码，采用big5繁体中文字标准进行编码
# 数据读入
train_data = pd.read_csv("train.csv",encoding = 'big5')
test_data = pd.read_csv("test.csv",encoding = 'big5')

# pandas进行数据处理
data = train_data.iloc[:,3:] # .csv文件从第3列开始的所有数据存放到data中，切片服从左闭右开
data[data == 'NR'] = 0       # 将RAINFALL的空数据用0进行填充
data_arr = data.to_numpy()   # data数据为<class 'pandas.core.frame.DataFrame'> 需要将其转换成数组的形式

# 特征提取
# 将数据从12*20*18（4320）*24 转换成 12*18*20*24（480）
month_data = {}           # 数据类型为字典
for month in range(12):   # range(n)函数 按顺序产生从0—(n-1)个数 
    sample = np.empty([18,480])
    for day in range(20):
        sample[:,day*24:(day+1)*24] = data_arr[18*(20*month+day):18*(20*month+day+1),:] 
    month_data[month] = sample
    
# 制作数据集，对数据进行标注
train = np.empty([471*12,18*9],dtype = float)
label = np.empty([471*12,1],dtype = float)

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour >14:
                continue
            train[month*471+day*24+hour,:] = month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,-1) 
                                                # reshape(n,-1)将数组转换成需要的n行，多少列无所谓
            label[month*471+day*24+hour,0] = month_data[month][9,day*24+hour+9]
            
# Normalize 归一化 原始数据有的很大，有的很小，进行feature scaling
# 求均值 axis = 0对各列求平均值
mean = np.mean(train,axis = 0)
std = np.std(train,axis = 0)
for i in range(len(train)):   # 对于二维数组a len(a)即得到二维数组的行
    for j in range((len(train[0]))):
        if std[j] != 0:
            train[i][j] = (train[i][j] - mean[j])/std[j]

# 设定训练集、交叉训练集、测试集
x_train_set = train[:math.floor(len(train)*0.8),:] # math.floor()求的结果是它的值小于或等于这个浮点数
y_train_set = label[:math.floor(len(label)*0.8),:]
x_validation = train[math.floor(len(train)*0.8):,:]
y_validation = label[math.floor(len(label)*0.8):,:]

print("len(x_train_set)",len(x_train_set))   # 4521
print("len(y_train_set)",len(y_train_set))   # 4521
print("len(x_validation)",len(x_validation)) # 1131
print("len(y_validation)",len(y_validation)) # 1131

# 训练模型  y = w * x + b
dim = 18*9+1           # 其中x为18*9 b为常数项
w = np.zeros([dim,1])  #用法：zeros(shape, dtype=float, order='C') 返回：返回来一个给定形状和类型的用0填充的数组
print('w,shape\n',w.shape) # w.shape = (163,1)

# np.concatenate() 用于数组的拼接 axis =1 数组的对应行进行拼接
# .astype() 用于类型转换
print("train.shape\n",train.shape)
x = np.concatenate((np.ones([12*471,1]),train),axis =1).astype(float) 
print("x.shape\n",x.shape)
print("x = ",x)
lr = 100
iter_time = 1000
adagrad = np.zeros([dim,1]) # 因为有163个参数需要更新，所以adagrad的维数是163个
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power((np.dot(x,w)-label),2))/471/12)
    if(t%100==0):
        print(str(t)+":"+str(loss))
    gradient = 2*np.dot(x.transpose(),np.dot(x,w)-label)
    # print("type)(gradient)",gradient.shape) # gradient.shape = (163,1)
    adagrad += gradient**2       # ** 代表乘方运算
    w = w-lr*gradient/np.sqrt(adagrad+eps)
np.save('weight.npy',w)

