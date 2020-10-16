# 预测PM2.5值
# 导入测试数据并对数据进行补充

test_data = pd.read_csv('/kaggle/input/ml2020spring-hw1/test.csv',header = None,encoding = 'big5')
# 在读入数据的时候没有 header = None的参数时，数据少一行！！

test_data = test_data.iloc[:,2:]
test_data[test_data == 'NR'] = 0
print(type(test_data)) # <class 'pandas.core.frame.DataFrame'> 此时导入的数据并不是数组，无法用数组的函数对其进行运算

test_data  = test_data.to_numpy()
print(type(test_data))
print("test_data shape\n",test_data.shape) # test_data.shape =(4319,9) 4319 * 9 = 38871
print("test_data",test_data)

# 构造测试集
test_x = np.empty([240,18*9],dtype = float) # 240 * 18 * 9 = 38880 
for day in range(240):
    test_x[day,:] = test_data[day*18:day*18+18,:].reshape(1,-1)

# 对测试数据进行归一化
mean_x  = np.mean(test_x,axis = 0)
std_x = np.std(test_x,axis = 0)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j]!=0:
            test_x[i][j] = (test_x[i][j]-mean_x[j]) / std_x[j]          
# print("test_x.shape",test_x.shape) test_x.shape (240, 162)
test_x = np.concatenate((np.ones([240,1]),test_x),axis = 1).astype(float)

# 读取权重并预测结果
W = np.load('weight.npy')
pre_y = np.dot(test_x,W)

# 将结果写入文件
import csv
with open('./submit.csv',mode = 'w',newline = '') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id','value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id' + str(i),pre_y[i][0]]
        csv_writer.writerow(row)
        print(row)
