# train processing head
import sys

from sklearn.externals import joblib
from time import clock

import pandas as pd
import numpy as np
import sklearn.metrics as skmet
from sklearn.metrics import accuracy_score
from pytorchtools import EarlyStopping
import torch
from torch import nn
from torch.nn import init
import time
import sampling


import os

def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]
    return positive, negative

def generate_valid_data(data, label, size=0.05):
    # 按照比例划分训练集 和 validation 集合
    positive_data, negative_data = divide_data(data, label)
    positive_length = positive_data.shape[0]
    negative_length = negative_data.shape[0]
    positive_label = np.ones(positive_length).reshape(-1, 1)
    negative_label = np.zeros(negative_length).reshape(-1, 1)

    positive_data_label = np.hstack((positive_data, positive_label))
    negative_data_label = np.hstack((negative_data, negative_label))

    np.random.shuffle(positive_data_label)
    np.random.shuffle(negative_data_label)

    positive_valid_length = max(int(positive_length * size), 1)
    negative_valid_length = max(int(negative_length * size), 1)
    
    valid_pos_data_label = positive_data_label[:positive_valid_length, :]
    train_pos_data_label = positive_data_label[positive_valid_length:, :]

    valid_neg_data_label = negative_data_label[:negative_valid_length, :]
    train_neg_data_label = negative_data_label[negative_valid_length:, :]

    valid_data_label = np.vstack((valid_pos_data_label, valid_neg_data_label))
    train_data_label = np.vstack((train_pos_data_label, train_neg_data_label))

    np.random.shuffle(valid_data_label)
    np.random.shuffle(train_data_label)

    valid_data = valid_data_label[:, :-1]
    valid_label = valid_data_label[:, -1].reshape(-1, 1)

    train_data = train_data_label[:, :-1]
    train_label = train_data_label[:, -1].reshape(-1, 1)

    return valid_data, valid_label, train_data, train_label



def get_select_data(source_data, batch_size):
    source_length = source_data.shape[0]
    select_index = np.random.choice(source_length, batch_size, replace=False)
    select_data = source_data[select_index]
    return select_data

def get_batch_size(batch_size, infor_method, positive_length, border_majority_length, informative_minority_length):
    if infor_method == 'normal':
        return min(positive_length, batch_size)

    if infor_method == 'im' or infor_method == 'im2' or infor_method == 'im3':
        return min(informative_minority_length, batch_size)
    
    if infor_method == 'bm':
        return min(border_majority_length, batch_size)
    
    if infor_method == 'both' or infor_method == 'both2' or infor_method == 'both3':
        return min(informative_minority_length, batch_size)
     



def generate_batch_data(train_data, train_label, batch_size):
    source_length = train_data.shape[0]
    batch_size = min(batch_size, source_length)
    select_index = np.random.choice(source_length, batch_size, replace=False)
    select_data = train_data[select_index]
    select_label = train_label[select_index]
    return select_data, select_label


def generate_train_data_label(pre_data, pos_data, ref_label, target_label):
    pre_length = pre_data.shape[0]
    pos_length = pos_data.shape[0]
    all_length = pre_length * pos_length

    # repeat 每一个都连续重复
    pre_repeat_data = np.repeat(pre_data, pos_length, axis=0)
    # tile 整体重复
    pos_tile_data = np.tile(pos_data, (pre_length, 1))

    ref_label_data = np.tile(ref_label, (all_length, 1)).reshape(-1, 1)

    pre_pos_label = np.tile(target_label, (all_length, 1)).reshape(-1,1)
    pre_pos_data = np.hstack((pre_repeat_data, pos_tile_data ))
    pre_pos_data = np.hstack((pre_pos_data, ref_label_data))

    pre_pos_data_label = np.hstack((pre_pos_data, pre_pos_label))
    return pre_pos_data_label



def transform_data_to_train_form(infor_method, train_data_Pos, train_data_neg, infor_Pos, border_neg):
    if infor_method == 'normal':
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]),  np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, train_data_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(train_data_Pos, train_data_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(train_data_Pos, train_data_Pos, np.array([1]), np.array([1]))

    if infor_method == 'bm':
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]), np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, train_data_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(train_data_Pos, train_data_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(train_data_Pos, train_data_Pos, np.array([1]), np.array([1]))

    if infor_method == 'im':
        # positive 全部替换为 informative positive
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]), np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, infor_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(infor_Pos, train_data_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(infor_Pos, infor_Pos, np.array([1]), np.array([1]))

    if infor_method == 'im2':
        # pp 对照组使用 informative positive
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]), np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, train_data_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(train_data_Pos, train_data_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(train_data_Pos, infor_Pos, np.array([1]), np.array([1]))

    if infor_method == 'im3':
        # 只有对照的sample 被替换为 informative positive
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]), np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, infor_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(train_data_Pos, train_data_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(train_data_Pos, infor_Pos, np.array([1]), np.array([1]))

    if infor_method == 'both':
        # positive 全部替换为 informative positive 对照组使用 border sample
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]), np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, infor_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(infor_Pos, train_data_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(infor_Pos, infor_Pos, np.array([1]), np.array([1]))
    
    if infor_method == 'both2':
        # nn 对照组使用 border sample， pp 对照组使用 informative positive
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]), np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, train_data_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(train_data_Pos, train_data_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(train_data_Pos, infor_Pos, np.array([1]), np.array([1]))
    
    if infor_method == 'both3':
        # 对照组全部换为 informative sample 和 border sample
        neg_neg_data_label = generate_train_data_label(train_data_neg, border_neg, np.array([0]), np.array([0]))
        neg_pos_data_label = generate_train_data_label(train_data_neg, infor_Pos, np.array([1]), np.array([0]))
        pos_neg_data_label = generate_train_data_label(train_data_Pos, border_neg, np.array([0]), np.array([1]))
        pos_pos_data_label = generate_train_data_label(train_data_Pos, infor_Pos, np.array([1]), np.array([1]))

    first_part_data_label = np.vstack((neg_neg_data_label, neg_pos_data_label))
    second_part_data_label = np.vstack((pos_neg_data_label, pos_pos_data_label))

    all_data_label = np.vstack((first_part_data_label, second_part_data_label))
    
    np.random.shuffle(all_data_label)

    transformed_data = all_data_label[:, :-1]
    transformed_label = all_data_label[:, -1:].reshape(-1, 1)

    return transformed_data, transformed_label



















def loadTrainData(file_name):
    file_data = np.loadtxt(file_name, delimiter=',')
    label = file_data[:,-1]
    data = np.delete(file_data, -1, axis=1)
    data = data.astype(np.float64)
    label = label.reshape(-1, 1)
    label = label.astype(np.int)
    return data, label

def get_infor_data(infor_method, train_data, train_label, positive_data, negative_data):
    if infor_method == 'normal':
        return positive_data, negative_data
    else:
        sampling_model = sampling.Sampling()
        border_majority_index, informative_minority_index = sampling_model.getTrainingSample(train_data, train_label)
        informative_minority_data = negative_data[border_majority_index]
        border_majority_data = positive_data[informative_minority_index]
        return informative_minority_data, border_majority_data


# def get_train_info(trian_method):
#     train_info_list = train_method.split('_')
#     model_type, infor_method, num_epochs = train_info_list
    


#     return model_type, sample_method, num_epochs





def set_para():
    global dataset_name
    global dataset_index
    global record_index
    global device_id
    global train_method

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'dataset_name':
            dataset_name = para[1]
        if para[0] == 'dataset_index':
            dataset_index = para[1]
        if para[0] == 'record_index':
            record_index = para[1]
        if para[0] == 'device_id':
            device_id = para[1]
        if para[0] == 'train_method':
            train_method = para[1]




# -------------------------------------parameters----------------------------------------
dataset_name = 'abalone19'
dataset_index = '1'
record_index = '1'
device_id = '1'
method_name = 'MLP_normal'
train_method = 'MLP_minus_notMirror_early'
num_epochs = 5000
batch_size = 100
start_epochs = 0
save_epoch = 2000
# ----------------------------------set parameters---------------------------------------
set_para()
train_file_name = './test_{0}/standlization_data/{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, train_method, record_index, dataset_index)


os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


# ----------------------------------start processing-------------------------------------
print(train_file_name)
print(model_name)
print('----------------------\n\n\n')



start = time.process_time()


base_train_data, base_train_label = loadTrainData(train_file_name)

# model_type, sample_method, early_stop_type, num_epochs = get_train_info(train_method)
model_type, infor_method, num_epochs = train_method.split('_')
num_epochs = int(num_epochs)


positive_data, negative_data = divide_data(base_train_data, base_train_label)
informative_minority_data, border_majority_data = get_infor_data(infor_method, base_train_data, base_train_label, positive_data, negative_data)


informative_minority_label = np.ones(informative_minority_data.shape[0]).reshape(-1, 1)
border_majority_label = np.zeros(border_majority_data.shape[0]).reshape(-1, 1)

informative_data_label = np.hstack((informative_minority_data, informative_minority_label))
border_majority_label = np.hstack((border_majority_data, border_majority_label))

infor_train_data_label = np.vstack((informative_data_label, border_majority_label))
np.shuffle(infor_train_data_label)

train_data = infor_train_data_label[:, :-1]
train_label = infor_train_data_label[:, -1].reshape(-1, 1)
input_dim = train_data.shape[1] * 2


patience = 20
# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)

class Classification(nn.Module):
    def __init__(self, input_dim):
        super(Classification, self).__init__()
        self.hidden_1 = nn.Linear(input_dim, 2*input_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2*input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden_1(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


dependency_dict = {
    20000: None,
    15000: 20000,
    10000: 15000,
    8000: 10000,
    5000: 8000,
    2000: 5000,
    0: 2000
}



# dependency_list = ['20000', '15000', '10000', '8000', '5000', '2000']
dependency_list = ['2000', '5000', '8000', '10000', '15000', '20000']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for train_epoch in dependency_list:
    history_train_method = 'MLP_{0}_{1}'.format(infor_method, train_epoch)
    history_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, history_train_method, record_index, dataset_index)
    # 判断追训模型是否存在
    print(history_model_name)
    if os.path.exists(history_model_name):
        # 找到模型，追训
        net = torch.load(history_model_name, map_location=device)
        start_epochs = int(train_epoch)
    else:
        # 没有找到，开始训练
        save_epoch = int(train_epoch)
        break

if start_epochs >= num_epochs:
    # 如果已经是最后一个，就直接退出
    print('running_time is 0')
    exit(0)

if start_epochs == 0:
    net = Classification(input_dim)
    init.normal_(net.hidden_1.weight, mean=0, std=0.01)
    init.normal_(net.output.weight, mean=0, std=0.01)
    init.constant_(net.hidden_1.bias, val=0)
    init.constant_(net.output.bias, val=0)
    
# 根据启动循环，确定推出位置
save_epoch = dependency_dict[start_epochs]

net.to(device)

loss_fn = nn.BCELoss()
loss_fn_1 = nn.BCELoss(reduction='none')
loss_fn_2 = nn.BCELoss(reduction='none')
loss_fn_3 = nn.BCELoss(reduction='none')
loss_fn_4 = nn.BCELoss(reduction='none')

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(net.parameters(), lr=1.2, momentum=0.9)

# input_valid_pre_data = torch.Tensor(torch.from_numpy(transformed_valid_pre_data).float())
# input_valid_pos_data = torch.Tensor(torch.from_numpy(transformed_valid_pos_data).float())
# input_valid_label = torch.Tensor(torch.from_numpy(transformed_valid_label).float())
# input_valid_pre_data = input_valid_pre_data.to(device)
# input_valid_pos_data = input_valid_pos_data.to(device)
# input_valid_label = input_valid_label.to(device)



# input_valid_data = torch.Tensor(torch.from_numpy(valid_data).float())
# input_valid_label = torch.Tensor(torch.from_numpy(valid_label).float())
# input_valid_data = input_valid_data.to(device)
# input_valid_label_gpu = input_valid_label.to(device)

cur_train_epochs = start_epochs
for epoch in range(num_epochs):
    if cur_train_epochs >= num_epochs:
        # 判断开始时是否已经是最新的位置
        break
    cur_train_epochs += 1


    train_x, train_y = generate_batch_data(train_data, train_label, batch_size)


    input_data = torch.Tensor(torch.from_numpy(train_x).float())
    input_label = torch.Tensor(torch.from_numpy(train_y).float())

    input_data = input_data.to(device)
    input_label = input_label.to(device)
    
    predict = net(input_data)

    loss = loss_fn(predict, input_label)
    # loss_1 = loss_fn_1(predict[:, 0], input_label[:,0])
    # loss_2 = loss_fn_2(predict[:, 1], input_label[:,1])
    # loss_3 = loss_fn_3(predict[:, 2], input_label[:,2])
    # loss_4 = loss_fn_4(predict[:, 3], input_label[:,3])
    # loss = torch.sum( loss_1 + loss_2 + loss_3 + loss_4 )

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.item()

    if epoch % 500 == 0:
        # valid_output = net(input_valid_data)
        # result =  torch.ge(valid_output, 0.5) 
        # result = result.cpu()
        # #计算准确率
        # train_acc = accuracy_score(input_valid_label, result)

        # #计算精确率
        # pre = skmet.precision_score(y_true=input_valid_label, y_pred=result)

        # #计算召回率
        # rec = skmet.recall_score(y_true=input_valid_label, y_pred=result)
        # f1 = skmet.f1_score(y_true=input_valid_label, y_pred=result)
        # auc = skmet.roc_auc_score(y_true=input_valid_label, y_score=result)
        # print('epoch {:.0f}, loss {:.4f}, train acc {:.2f}%, f1 {:.4f}, precision {:.4f}, recall {:.4f}, auc {:.4f}'.format(epoch+1, train_loss, train_acc*100, f1, pre, rec, auc) )
        print('epoch {:.0f}, loss {:.4f}'.format(epoch+1, train_loss) )
    if cur_train_epochs == save_epoch:
        cur_train_method = 'MLP_{0}_{1}'.format(infor_method, str(save_epoch))
        cur_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, cur_train_method, record_index, dataset_index)
        torch.save(net, cur_model_name)
        save_epoch = dependency_dict[cur_train_epochs]
        print('save model {0}'.format(cur_model_name))

        last_train_epoch = 0
        for train_epoch in dependency_list:
            history_train_method = 'MLP_{0}_{1}'.format(infor_method, str(train_epoch))
            history_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, history_train_method, record_index, dataset_index)
            # 判断追训模型是否存在
            print(history_model_name)
            if os.path.exists(history_model_name):
                # 更新已有模型标记
                last_train_epoch = train_epoch
            else:
                save_epoch = int(train_epoch)
                if last_train_epoch != cur_train_epochs:
                    history_train_method = 'MLP_{0}_{1}'.format(infor_method, str(last_train_epoch))
                    history_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, history_train_method, record_index, dataset_index)
                    net = torch.load(history_model_name, map_location=device)
                    cur_train_epochs = int(last_train_epoch)
                break
    
    if cur_train_epochs >= num_epochs:
        break

        

        
cur_train_method_list = [model_type, infor_method, str(save_epoch)]
cur_train_method = '_'.join(cur_train_method_list)
cur_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, cur_train_method, record_index, dataset_index)
torch.save(net, cur_model_name)
print('save model {0}'.format(cur_model_name))   

torch.save(net, model_name)


finish = time.process_time()
running_time = finish-start
print('running_time is {0}'.format(running_time))
