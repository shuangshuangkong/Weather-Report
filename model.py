import pandas as pd
import matplotlib.pyplot as plt

ett_h1=pd.read_csv('data/jena_climate_2009_2016.csv',header=0)
ett_h1_columns=list(ett_h1.columns)
ett_h1_date=ett_h1['Date Time']
#l=int(len(ett_h1_date)/15)
#ett_h1_date_label = [str(ii) for ii in ett_h1_date]
'''
plt.plot(ett_h1_date,ett_h1['OT'])
plt.show()'''

plt.figure(figsize=(11,10))
for i in range(1,len(ett_h1_columns)):
    plt.subplot(len(ett_h1_columns)-1,1,i)
    plt.plot(ett_h1_date,ett_h1[ett_h1_columns[i]])
    #plt.xticks(ett_h1_date_label[::l],ett_h1_date[::l],fontsize=8,rotation=15)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout()
plt.show()



import os
import paddle
import paddle.fluid as fluid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data visualization
def show_raw_visualization(data):
    time_data = data['Date Time']
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(ax=axes[i // 2, i % 2],title=key,rot=25,)
        ax.legend(key)
    plt.tight_layout()
# data normalization
def normalization(data):
    avg = np.mean(data, axis=0)#axis=0表示按数组元素的列对numpy取相关操作值
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)
    result_data = (data - avg) / (max_ - min_)
    return result_data

#################################################
# read data
data = pd.read_csv("./data/jena_climate_2009_2016.csv", header=0)
# data visualization
feature_keys = list(data.columns)[1:]
# show_raw_visualization(data)

# data preprocessing
y = data[feature_keys[1]] # Temperature in Celsius
X = data[feature_keys[2:]]
X.insert(0,feature_keys[0],data[feature_keys[0]])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

train_data=pd.concat([x_train,y_train],axis=1)
test_data=pd.concat([x_test,y_test],axis=1)
train_data = normalization(train_data.values)
test_data = normalization(test_data.values)

def my_train_reader():
    def reader():
        for temp in train_data:
            yield temp[:-1], temp[-1]
    return reader
 
def my_test_reader():
    def reader():
        for temp in test_data:
            yield temp[:-1], temp[-1]    
    return reader
# 定义batch
train_reader = paddle.batch(
    my_train_reader(),
    batch_size=10)


# model prepare
paddle.enable_static()
DIM = 1
hid_dim2 = 1
x = fluid.layers.data(name='x',shape=[DIM],dtype='float32',lod_level=1)
label = fluid.layers.data(name='y',shape=[1],dtype='float32')

fc0=fluid.layers.fc(input=x,size=DIM*4)
lstm_h,c=fluid.layers.dynamic_lstm(input=fc0,size=DIM*4,is_reverse=False)
lstm_max=fluid.layers.sequence_pool(input=lstm_h,pool_type='max')
lstm_max_tanh=fluid.layers.tanh(lstm_max)
prediction=fluid.layers.fc(input=lstm_max_tanh,size=hid_dim2,act='tanh')
cost=fluid.layers.square_error_cost(input=prediction,label=label)
avg_cost=fluid.layers.mean(x=cost)


adam_optimizer=fluid.optimizer.Adam(learning_rate=0.001)
adam_optimizer.minimize(avg_cost)
place=fluid.CPUPlace()
exe=fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder=fluid.DataFeeder(place=place,feed_list=[x,label])
# train prepare
# 定义双层循环
def train_loop():
    step = 0 
    PASS_NUM = 100
    for pass_id in range(PASS_NUM):
        total_loss_pass = 0
        for data in train_reader():        
            avg_loss_value, = exe.run(
                fluid.default_main_program(), 
                feed= feeder.feed(data), 
                fetch_list=[avg_cost])
            total_loss_pass += avg_loss_value
#         print("Pass %d, total avg cost = %f" % (pass_id, total_loss_pass))
        # 画图
        plot_cost.append(train_title, step, avg_loss_value)
        step += 1
        plot_cost.plot()
    fluid.io.save_inference_model(SAVE_DIRNAME, ['x'], [prediction], exe)
    
    
    
# test prepare
def convert2LODTensor(temp_arr, len_list):
    temp_arr = np.array(temp_arr) 
    temp_arr = temp_arr.flatten().reshape((-1, 1))
    print(temp_arr.shape)
    return fluid.create_lod_tensor(
        data=temp_arr,
        recursive_seq_lens =[len_list],
        place=fluid.CPUPlace()
        )
    
 
def get_tensor_label(mini_batch):  
    tensor = None
    labels = []
    
    temp_arr = []
    len_list = []
    for _ in mini_batch: 
        labels.append(_[1]) 
        temp_arr.append(_[0]) 
        len_list.append(len(_[0])) 
    tensor = convert2LODTensor(temp_arr, len_list)    
    return tensor, labels
 
my_tensor = None
labels = None
 
# 定义batch
test_reader = paddle.batch(
    my_test_reader(),
    batch_size=325)
 
for mini_batch in test_reader():
    my_tensor,labels = get_tensor_label(mini_batch)
    break
place = fluid.CPUPlace()
exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets] = (
        fluid.io.load_inference_model(SAVE_DIRNAME, exe))
    results = exe.run(inference_program,
                      feed= {'x': my_tensor}, 
                      fetch_list=fetch_targets)

result_print = results[0].flatten()
plt.figure()
plt.plot(list(range(len(labels))), labels, color='r')  #红线为真实值
plt.plot(list(range(len(result_print))), result_print, color='g')  #绿线为预测值
plt.show()
