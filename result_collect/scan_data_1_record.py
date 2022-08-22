# -*- coding: utf-8 -*-

# import glob
import re
import sys
import numpy as np
# import matplotlib
# matplotlib.use('agg')
# from matplotlib import pyplot as plt
import csv
import copy
import pandas as pd


class data_record_collect:
    '''
        读取结果文件夹，整理排序，生成\t分割的输出 string
    '''
    def __init__(self, dataset_path, file_name_prefix, method, train_method, test_method, data_range=5, offset=1):
        '''
            初始化，确定dataset_path等信息，以及偏移量和数据数量
        '''
        # self.auc_list = {}
        # self.f1_list = {}
        # self.precision_list = {}
        # self.recall = {}
        self.score_record = {}
        self.file_name_prefix = file_name_prefix
        self.method = method
        self.train_method = train_method
        self.test_method = test_method
        self.data_range = data_range
        self.offset = offset
        self.dataset_path = dataset_path
    
    def scan_file(self, file_name, data_index):
        '''
            读取文件数据，获取 [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        '''
        # auc_str = 'the AUC is '
        # f_score_str = 'the Fscore is '
        # pre_str = 'the precision is '
        # recall_str = 'the recall is '
        # print(file_name)
        # print(data_index)
        try:
            # 当文件存在时
            with open(file_name,'r') as fr:
                for line in fr:
                    # 解析字符串，获取指标数据
                    line = line.replace('the ', '')
                    score_type, score_value = line.split(' is ')
                    # print(score_type, score_value)
                    # 获取指标字典，按照index存储
                    cur_score_record = self.score_record.get(score_type, {})
                    cur_score_record[data_index] = float(score_value)
                    self.score_record[score_type] = cur_score_record
        except:
            print('error')
            # 文件不存在时跳过
            pass

    def scan_all_file(self):
        '''扫描文件夹路径下所有文件'''
        for data_index in range(self.offset, self.offset+self.data_range):
            cur_file_name = self.dataset_path + self.file_name_prefix + '_{0}_pred_result.txt'.format(data_index)
            self.scan_file(cur_file_name, data_index)
    
    def get_all_valid_value(self):
        '''获取所有有效数据'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        if self.score_record != {}:
            for score_type in self.score_record:
                cur_score_record_dict = self.score_record[score_type]
                cur_all_result_value = []
                # 通过数据文件的范围进行检查
                for data_index in range(self.offset, self.offset+self.data_range):
                    if data_index in cur_score_record_dict:
                        cur_all_result_value.append(cur_score_record_dict[data_index])
                cur_score_record_dict['all_vaild_value_list'] = cur_all_result_value
                self.score_record[score_type] = cur_score_record_dict
        else:
            default_empty_dict = {}
            default_empty_dict['all_vaild_value_list'] = [0 for i in range(self.data_range)]
            for score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ]:
                self.score_record[score_type] = default_empty_dict


    def get_all_value(self):
        '''获取所有有效数据，没有的部分补 -1'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        for score_type in self.score_record:
            
            cur_score_record_dict = self.score_record[score_type]
            cur_sorted_value = [-1 for i in range(self.data_range)]
            for data_index in range(self.offset, self.offset+self.data_range):
                if data_index in cur_score_record_dict:
                    # 如果存在
                    cur_sorted_value[data_index-self.offset] = cur_score_record_dict[data_index]
                    # 如果不存在， 上边已经写成 -1 了
            cur_score_record_dict['all_value_list'] = cur_sorted_value
            self.score_record[score_type] = cur_score_record_dict
        



    def get_avgerage_value(self):
        '''获取平均值，并对不存在的数据补 -1，生成排序后的list 单独存储'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valid_value = cur_score_record_dict['all_vaild_value_list']
            average_value = -1
            if len(cur_all_valid_value) != 0:
                average_value = float(sum(cur_all_valid_value)) / len(cur_all_valid_value)
            cur_score_record_dict['average_value'] = average_value
            self.score_record[score_type] = cur_score_record_dict
    

    def get_max(self):
        '''获取结果最大值，并保存'''
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valid_value = cur_score_record_dict['all_vaild_value_list']
            max_value = -1
            if len(cur_all_valid_value) != 0:
                max_value = max(cur_all_valid_value)
            cur_score_record_dict['max_value'] = max_value
            self.score_record[score_type] = cur_score_record_dict
    
    def get_min(self):
        '''获取结果最小值，并保存'''
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valid_value = cur_score_record_dict['all_vaild_value_list']
            min_value = -1
            if len(cur_all_valid_value) != 0:
                min_value = min(cur_all_valid_value)
            cur_score_record_dict['min_value'] = min_value
            self.score_record[score_type] = cur_score_record_dict
    
    def get_amm_value(self):
        '''获取平均值，最大值，最小值'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valid_value = cur_score_record_dict['all_vaild_value_list']
            # print(score_type)
            # print(cur_all_valid_value)
            average_value = -1
            max_value = -1
            min_value = -1
            if len(cur_all_valid_value) != 0:
                average_value = float(sum(cur_all_valid_value)) / len(cur_all_valid_value)
                max_value = max(cur_all_valid_value)
                min_value = min(cur_all_valid_value)
            cur_score_record_dict['average_value'] = average_value
            cur_score_record_dict['max_value'] = max_value
            cur_score_record_dict['min_value'] = min_value
            self.score_record[score_type] = cur_score_record_dict
    
    def get_print_str(self):
        '''按照顺序生成输出字符串，顺序是平均Fscore，最大，最小，以及全部数据'''
        self.scan_all_file() 
        self.get_all_valid_value()
        self.get_all_value()
        self.get_amm_value()

        score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]

        all_sorted_value = []
        for score_type in score_type_list:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_value_list = cur_score_record_dict['all_value_list']
            all_sorted_value.append(cur_score_record_dict['average_value'])
            all_sorted_value.append(cur_score_record_dict['max_value'])
            all_sorted_value.append(cur_score_record_dict['min_value'])
            all_sorted_value += cur_all_value_list
        all_sorted_value = list(map(str, all_sorted_value))
        # print(self.file_name_prefix)
        # print(self.method)
        cur_train_test_info = self.method.split('_')
        self.output_str = '{0}_{1}'.format(self.file_name_prefix, self.method) + '\t' + '\t'.join(cur_train_test_info) + '\t' + '\t'.join( all_sorted_value )
        return self.output_str
    
    def get_all_metrix_data(self):
        '''按照顺序生成输出字符串，顺序是平均Fscore，最大，最小，以及全部数据'''
        self.scan_all_file() 
        self.get_all_valid_value()
        self.get_all_value()
        self.get_amm_value()

        score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]

        all_sorted_value = []

        for score_type in score_type_list:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_value_list = cur_score_record_dict['all_value_list']
            all_sorted_value.append(cur_score_record_dict['average_value'])
            all_sorted_value.append(cur_score_record_dict['max_value'])
            all_sorted_value.append(cur_score_record_dict['min_value'])
            all_sorted_value += cur_all_value_list
        # all_sorted_value = list(map(str, all_sorted_value))
        # print(self.file_name_prefix)
        # print(self.method)
        all_value = []
        cur_train_test_info = self.method.split('_')
        all_value.append( '{0}_{1}'.format(self.file_name_prefix, self.method) )
        # all_value += [ self.train_method, self.test_method ]
        all_value += cur_train_test_info
        all_value += all_sorted_value
        self.all_value = all_value
        return self.all_value


    def get_summary_print_str(self):
        '''按照顺序生成输出字符串，顺序是平均Fscore，最大，最小，以及全部数据'''
        self.scan_all_file() 
        self.get_all_valid_value()
        self.get_all_value()
        self.get_amm_value()

        score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]

        all_sorted_value = []
        for score_type in score_type_list:
            cur_score_record_dict = self.score_record[score_type]
            # cur_all_value_list = cur_score_record_dict['all_value_list']
            all_sorted_value.append(cur_score_record_dict['average_value'])
            all_sorted_value.append(cur_score_record_dict['max_value'])
            all_sorted_value.append(cur_score_record_dict['min_value'])
            # all_sorted_value += cur_all_value_list
        all_sorted_value = list(map(str, all_sorted_value))
        # print(self.file_name_prefix)
        # print(self.method)
        self.output_str = '{0}_{1}'.format(self.file_name_prefix, self.method) + '\t' + '\t'.join( all_sorted_value )
        return self.output_str




# dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']
# dataset_list = ['abalone19']

dataset_list = ['pima','glass0','vehicle0','ecoli1','yeast3','pageblocks1','glass5','yeast5','yeast6','abalone19']

data_range = 5
record_index = 1

train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# early_stop_type_list = [ '20000', '15000', '10000', '8000']
# early_stop_type_list = [ '5000', '2000']
# early stop 效果不太明显， 结果不太好
test_infor_method_list = [ 'normal', 'bm', 'im', 'both']
# test_infor_method_list = [ 'normal', 'bm', 'im', 'both']

ref_num_type_list = ['num']
ref_times_list = ['10']
boundary_type_list = ['half']

# transform_list = ['concat', 'minus']
# mirror_type_list = ['Mirror', 'notMirror']
# early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# early stop 效果不太明显， 结果不太好
score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]


writer = pd.ExcelWriter("test_result_all_main_info.xlsx")

# 建立表头
first_line_info = ['method_name', 'model_type', 'train_sample_method', 'train_epoch', 'test_info_method', 'ref_num_type', 'ref_times', 'boundary_type']

for score_type in score_type_list:
    cur_part_first_line_info = []
    cur_part = ''

    cur_part_first_line_info.append( '{0}_average_value'.format(score_type) )
    cur_part_first_line_info.append( '{0}_max_value'.format(score_type) )
    cur_part_first_line_info.append( '{0}_min_value'.format(score_type) )


    for index in range(1, 6):
        cur_part_first_line_info.append( score_type + 'all_value_{0}'.format(index) )


    first_line_info += cur_part_first_line_info

select_line_info = ['model_type', 'train_sample_method', 'train_epoch', 'test_info_method', 'ref_num_type', 'ref_times', 'boundary_type']



for score_type in score_type_list:
    cur_part_first_line_info = []
    cur_part = ''

    cur_part_first_line_info.append( '{0}_average_value'.format(score_type) )
    cur_part_first_line_info.append( '{0}_max_value'.format(score_type) )
    cur_part_first_line_info.append( '{0}_min_value'.format(score_type) )

    select_line_info += cur_part_first_line_info

all_line_info_pd_list = []
select_line_info_pd_list = []
max_info_pd_list = []
dataset_max_info_dict = {}
all_line_info_pd_dict = {}

score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]
for dataset in dataset_list:
    all_line_info = []

    for sample_method in train_infor_method_list:
        for test_infor_method in test_infor_method_list:
            cur_train_test_type = '_'.join([sample_method, test_infor_method])

            for early_stop_type in early_stop_type_list:
                for ref_num_type in ref_num_type_list:
                    for ref_times in ref_times_list:
                        for boundary_type in boundary_type_list:
                            cur_command_list = []
                            train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
                            test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
                            cur_dataset_path = './test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
                            cur_file_name_prefix = dataset
                            cur_method = '{0}_{1}'.format(train_method, test_method)
                            
                            print(cur_dataset_path)
                            cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method, train_method=train_method, test_method=test_method)
                            cur_line = cur_obj.get_all_metrix_data()
                            all_line_info.append(cur_line)
                            # cur_loss_pro_list.append(cur_obj)
                            cur_type_targe_obj_dict = all_line_info_pd_dict.get(cur_train_test_type, {})

                            # 获取最大值
                            if cur_type_targe_obj_dict == {}:
                                cur_type_targe_obj_dict['max_Fscore'] = cur_obj
                                cur_type_targe_obj_dict['max_AUC'] = cur_obj
                            else:
                                cur_max_f_obj = cur_type_targe_obj_dict['max_Fscore']
                                cur_max_a_obj = cur_type_targe_obj_dict['max_AUC']

                                if cur_obj.score_record['Fscore']['average_value'] > cur_max_f_obj.score_record['Fscore']['average_value']:
                                    cur_type_targe_obj_dict['max_Fscore'] = cur_obj
                                
                                if cur_obj.score_record['AUC']['average_value'] > cur_max_f_obj.score_record['AUC']['average_value']:
                                    cur_type_targe_obj_dict['max_AUC'] = cur_obj
                            all_line_info_pd_dict[cur_train_test_type] = cur_type_targe_obj_dict

    
    all_line_info_pd = pd.DataFrame(all_line_info)
    all_line_info_pd.columns = first_line_info

    all_line_info_pd = all_line_info_pd.set_index('method_name')
    sheet_name = '{0}_all_metrix_info'.format(dataset)
    all_line_info_pd.style.background_gradient(subset=['Fscore_average_value', 'AUC_average_value'],cmap='Reds',vmin=0.0,vmax=1.0)
    all_line_info_pd_list.append( (sheet_name, all_line_info_pd) )
    all_line_info_pd.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1,0) )

    select_data_pd = all_line_info_pd[select_line_info]

    sheet_name = '{0}_main_metrix_info'.format(dataset)
    select_data_pd.style.background_gradient(subset=['Fscore_average_value', 'AUC_average_value'],cmap='Reds',vmin=0.0,vmax=1.0)
    select_line_info_pd_list.append( (sheet_name, select_data_pd))
    select_data_pd.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1,0) )                     


    max_info_list = []
    
    for sample_method in train_infor_method_list:
        for test_infor_method in test_infor_method_list:
            cur_train_test_type = '_'.join([sample_method, test_infor_method])
            cur_type_targe_obj_dict = all_line_info_pd_dict.get(cur_train_test_type, {})
            cur_max_f_obj = cur_type_targe_obj_dict['max_Fscore']
            cur_max_a_obj = cur_type_targe_obj_dict['max_AUC']
            
            cur_line = cur_max_f_obj.get_all_metrix_data()
            # cur_line.append('Fscore')
            max_info_list.append(cur_line)

            # cur_line = cur_max_a_obj.get_all_metrix_data()
            # cur_line.append('AUC')
            # max_info_list.append(cur_line)
    # first_line_info.append('matrix_type')
    max_info_list_pd = pd.DataFrame(max_info_list)
    max_info_list_pd.columns = first_line_info
    max_info_list_pd = max_info_list_pd.set_index('method_name')
    select_max_data_pd = max_info_list_pd[select_line_info]
    
    sheet_name = '{0}_max_matrix_info'.format(dataset)
    select_max_data_pd.style.background_gradient(subset=['Fscore_average_value', 'AUC_average_value'],cmap='Reds',vmin=0.0,vmax=1.0)
    
    select_max_data_pd.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1,0) )
    max_info_pd_list.append( (sheet_name, select_max_data_pd))
    dataset_max_info_dict[dataset] = all_line_info_pd_dict
    all_line_info_pd_dict = {}

writer.save()
writer.close()


writer = pd.ExcelWriter("test_result_all_info.xlsx")
for sheet_name, write_pd in all_line_info_pd_list:
    write_pd.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1,0) )

writer.save()
writer.close()


writer = pd.ExcelWriter("test_result_main_info.xlsx")
for sheet_name, write_pd in select_line_info_pd_list:
    write_pd.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1,0) )

writer.save()
writer.close()

writer = pd.ExcelWriter("test_result_max_info.xlsx")
for sheet_name, write_pd in max_info_pd_list:
    write_pd.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1,0) )

writer.save()
writer.close()


total_record = ''
first_line = []
for dataset in dataset_list:
    first_line.append(dataset)

normal_normal_list = []
normal_bm_list = []
normal_im_list = []
normal_both_list = []
bm_normal_list = []
bm_bm_list = []
bm_im_list = []
bm_both_list = []
im_normal_list = []
im_bm_list = []
im_im_list = []
im_both_list = []
im2_normal_list = []
im2_bm_list = []
im2_im_list = []
im2_both_list = []
im3_normal_list = []
im3_bm_list = []
im3_im_list = []
im3_both_list = []
both_normal_list = []
both_bm_list = []
both_im_list = []
both_both_list = []
both2_normal_list = []
both2_bm_list = []
both2_im_list = []
both2_both_list = []
both3_normal_list = []
both3_bm_list = []
both3_im_list = []
both3_both_list = []

all_line = [first_line]
for dataset in dataset_list:
    cur_dataset_max_info_dict = dataset_max_info_dict[dataset]

    normal_normal_obj = cur_dataset_max_info_dict['normal_normal']['max_Fscore']
    normal_bm_obj = cur_dataset_max_info_dict['normal_bm']['max_Fscore']
    normal_im_obj = cur_dataset_max_info_dict['normal_im']['max_Fscore']
    normal_both_obj = cur_dataset_max_info_dict['normal_both']['max_Fscore']
    bm_normal_obj = cur_dataset_max_info_dict['bm_normal']['max_Fscore']
    bm_bm_obj = cur_dataset_max_info_dict['bm_bm']['max_Fscore']
    bm_im_obj = cur_dataset_max_info_dict['bm_im']['max_Fscore']
    bm_both_obj = cur_dataset_max_info_dict['bm_both']['max_Fscore']
    im_normal_obj = cur_dataset_max_info_dict['im_normal']['max_Fscore']
    im_bm_obj = cur_dataset_max_info_dict['im_bm']['max_Fscore']
    im_im_obj = cur_dataset_max_info_dict['im_im']['max_Fscore']
    im_both_obj = cur_dataset_max_info_dict['im_both']['max_Fscore']
    im2_normal_obj = cur_dataset_max_info_dict['im2_normal']['max_Fscore']
    im2_bm_obj = cur_dataset_max_info_dict['im2_bm']['max_Fscore']
    im2_im_obj = cur_dataset_max_info_dict['im2_im']['max_Fscore']
    im2_both_obj = cur_dataset_max_info_dict['im2_both']['max_Fscore']
    im3_normal_obj = cur_dataset_max_info_dict['im3_normal']['max_Fscore']
    im3_bm_obj = cur_dataset_max_info_dict['im3_bm']['max_Fscore']
    im3_im_obj = cur_dataset_max_info_dict['im3_im']['max_Fscore']
    im3_both_obj = cur_dataset_max_info_dict['im3_both']['max_Fscore']
    both_normal_obj = cur_dataset_max_info_dict['both_normal']['max_Fscore']
    both_bm_obj = cur_dataset_max_info_dict['both_bm']['max_Fscore']
    both_im_obj = cur_dataset_max_info_dict['both_im']['max_Fscore']
    both_both_obj = cur_dataset_max_info_dict['both_both']['max_Fscore']
    both2_normal_obj = cur_dataset_max_info_dict['both2_normal']['max_Fscore']
    both2_bm_obj = cur_dataset_max_info_dict['both2_bm']['max_Fscore']
    both2_im_obj = cur_dataset_max_info_dict['both2_im']['max_Fscore']
    both2_both_obj = cur_dataset_max_info_dict['both2_both']['max_Fscore']
    both3_normal_obj = cur_dataset_max_info_dict['both3_normal']['max_Fscore']
    both3_bm_obj = cur_dataset_max_info_dict['both3_bm']['max_Fscore']
    both3_im_obj = cur_dataset_max_info_dict['both3_im']['max_Fscore']
    both3_both_obj = cur_dataset_max_info_dict['both3_both']['max_Fscore']

    normal_normal_list.append(normal_normal_obj.score_record['Fscore']['average_value'])
    normal_bm_list.append(normal_bm_obj.score_record['Fscore']['average_value'])
    normal_im_list.append(normal_im_obj.score_record['Fscore']['average_value'])
    normal_both_list.append(normal_both_obj.score_record['Fscore']['average_value'])
    bm_normal_list.append(bm_normal_obj.score_record['Fscore']['average_value'])
    bm_bm_list.append(bm_bm_obj.score_record['Fscore']['average_value'])
    bm_im_list.append(bm_im_obj.score_record['Fscore']['average_value'])
    bm_both_list.append(bm_both_obj.score_record['Fscore']['average_value'])
    im_normal_list.append(im_normal_obj.score_record['Fscore']['average_value'])
    im_bm_list.append(im_bm_obj.score_record['Fscore']['average_value'])
    im_im_list.append(im_im_obj.score_record['Fscore']['average_value'])
    im_both_list.append(im_both_obj.score_record['Fscore']['average_value'])
    im2_normal_list.append(im2_normal_obj.score_record['Fscore']['average_value'])
    im2_bm_list.append(im2_bm_obj.score_record['Fscore']['average_value'])
    im2_im_list.append(im2_im_obj.score_record['Fscore']['average_value'])
    im2_both_list.append(im2_both_obj.score_record['Fscore']['average_value'])
    im3_normal_list.append(im3_normal_obj.score_record['Fscore']['average_value'])
    im3_bm_list.append(im3_bm_obj.score_record['Fscore']['average_value'])
    im3_im_list.append(im3_im_obj.score_record['Fscore']['average_value'])
    im3_both_list.append(im3_both_obj.score_record['Fscore']['average_value'])
    both_normal_list.append(both_normal_obj.score_record['Fscore']['average_value'])
    both_bm_list.append(both_bm_obj.score_record['Fscore']['average_value'])
    both_im_list.append(both_im_obj.score_record['Fscore']['average_value'])
    both_both_list.append(both_both_obj.score_record['Fscore']['average_value'])
    both2_normal_list.append(both2_normal_obj.score_record['Fscore']['average_value'])
    both2_bm_list.append(both2_bm_obj.score_record['Fscore']['average_value'])
    both2_im_list.append(both2_im_obj.score_record['Fscore']['average_value'])
    both2_both_list.append(both2_both_obj.score_record['Fscore']['average_value'])
    both3_normal_list.append(both3_normal_obj.score_record['Fscore']['average_value'])
    both3_bm_list.append(both3_bm_obj.score_record['Fscore']['average_value'])
    both3_im_list.append(both3_im_obj.score_record['Fscore']['average_value'])
    both3_both_list.append(both3_both_obj.score_record['Fscore']['average_value'])

all_line.append(normal_normal_list)
all_line.append(normal_bm_list)
all_line.append(normal_im_list)
all_line.append(normal_both_list)
all_line.append(bm_normal_list)
all_line.append(bm_bm_list)
all_line.append(bm_im_list)
all_line.append(bm_both_list)
all_line.append(im_normal_list)
all_line.append(im_bm_list)
all_line.append(im_im_list)
all_line.append(im_both_list)
all_line.append(im2_normal_list)
all_line.append(im2_bm_list)
all_line.append(im2_im_list)
all_line.append(im2_both_list)
all_line.append(im3_normal_list)
all_line.append(im3_bm_list)
all_line.append(im3_im_list)
all_line.append(im3_both_list)
all_line.append(both_normal_list)
all_line.append(both_bm_list)
all_line.append(both_im_list)
all_line.append(both_both_list)
all_line.append(both2_normal_list)
all_line.append(both2_bm_list)
all_line.append(both2_im_list)
all_line.append(both2_both_list)
all_line.append(both3_normal_list)
all_line.append(both3_bm_list)
all_line.append(both3_im_list)
all_line.append(both3_both_list)

all_str = ''
for item_list in all_line:
    item_str_list = list(map(str, item_list))
    item_str = '\t'.join(item_str_list) + '\n'

    all_str += item_str


with open('result_table_1.txt','w') as record:
    # record.write(table_1)
    record.write(all_str)


    # table_2 = '\\section{{ results of {0} }}\n'.format(dataset)
    # table_2 += '\\begin{table}[H]\n'
    # table_2 += '\\centering\n'
    # table_2 += '\\caption{the performance of different varienties of ijcai method (AUC and F1}\n'
    # table_2 += '\\label{tab:ChangingTrainData33}\n'
    # table_2 += '\\begin{tabular}{|p{0.1\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|}\n'
    # table_2 += '\\hline \multirow{2}{*}{Method} & \multicolumn{2}{|c|}{Original test} & \multicolumn{2}{|c|}{Border Majority} & \multicolumn{2}{|c|}{Informative Minority} & \multicolumn{2}{|c|}{Both of them}\\\\\n'
    # table_2 += '\\cline{2-9} & Auc & F1  & Auc & F1 & Auc & F1 & Auc & F1  \\\\\n'

    # table_2 += '\\hline IJCAI & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    #     normal_normal_obj.score_record['AUC']['average_value'], normal_normal_obj.score_record['Fscore']['average_value'], 
    #     normal_bm_obj.score_record['AUC']['average_value'], normal_bm_obj.score_record['Fscore']['average_value'],
    #     normal_im_obj.score_record['AUC']['average_value'], normal_im_obj.score_record['Fscore']['average_value'],
    #     normal_both_obj.score_record['AUC']['average_value'], normal_both_obj.score_record['Fscore']['average_value'])

    # table_2 += '\\hline Training with BM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    # bm_normal_obj.score_record['AUC']['average_value'], bm_normal_obj.score_record['Fscore']['average_value'], 
    # bm_bm_obj.score_record['AUC']['average_value'], bm_bm_obj.score_record['Fscore']['average_value'],
    # bm_im_obj.score_record['AUC']['average_value'], bm_im_obj.score_record['Fscore']['average_value'],
    # bm_both_obj.score_record['AUC']['average_value'], bm_both_obj.score_record['Fscore']['average_value'])

    # table_2 += '\\hline Training with IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    # im_normal_obj.score_record['AUC']['average_value'], im_normal_obj.score_record['Fscore']['average_value'], 
    # im_bm_obj.score_record['AUC']['average_value'], im_bm_obj.score_record['Fscore']['average_value'],
    # im_im_obj.score_record['AUC']['average_value'], im_im_obj.score_record['Fscore']['average_value'],
    # im_both_obj.score_record['AUC']['average_value'], im_both_obj.score_record['Fscore']['average_value'])

    # table_2 += '\\hline Training with IM2 & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    # im2_normal_obj.score_record['AUC']['average_value'], im2_normal_obj.score_record['Fscore']['average_value'], 
    # im2_bm_obj.score_record['AUC']['average_value'], im2_bm_obj.score_record['Fscore']['average_value'],
    # im2_im_obj.score_record['AUC']['average_value'], im2_im_obj.score_record['Fscore']['average_value'],
    # im2_both_obj.score_record['AUC']['average_value'], im2_both_obj.score_record['Fscore']['average_value'])

    # table_2 += '\\hline Training with IM3 & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    # im3_normal_obj.score_record['AUC']['average_value'], im3_normal_obj.score_record['Fscore']['average_value'], 
    # im3_bm_obj.score_record['AUC']['average_value'], im3_bm_obj.score_record['Fscore']['average_value'],
    # im3_im_obj.score_record['AUC']['average_value'], im3_im_obj.score_record['Fscore']['average_value'],
    # im3_both_obj.score_record['AUC']['average_value'], im3_both_obj.score_record['Fscore']['average_value'])

    # table_2 += '\\hline Training wiht BM and IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    # both_normal_obj.score_record['AUC']['average_value'], both_normal_obj.score_record['Fscore']['average_value'], 
    # both_bm_obj.score_record['AUC']['average_value'], both_bm_obj.score_record['Fscore']['average_value'],
    # both_im_obj.score_record['AUC']['average_value'], both_im_obj.score_record['Fscore']['average_value'],
    # both_both_obj.score_record['AUC']['average_value'], both_both_obj.score_record['Fscore']['average_value'])

    # table_2 += '\\hline Training wiht BM and IM2 & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    # both2_normal_obj.score_record['AUC']['average_value'], both2_normal_obj.score_record['Fscore']['average_value'], 
    # both2_bm_obj.score_record['AUC']['average_value'], both2_bm_obj.score_record['Fscore']['average_value'],
    # both2_im_obj.score_record['AUC']['average_value'], both2_im_obj.score_record['Fscore']['average_value'],
    # both2_both_obj.score_record['AUC']['average_value'], both2_both_obj.score_record['Fscore']['average_value'])

    # table_2 += '\\hline Training wiht BM and IM3 & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(
    # both3_normal_obj.score_record['AUC']['average_value'], both3_normal_obj.score_record['Fscore']['average_value'], 
    # both3_bm_obj.score_record['AUC']['average_value'], both3_bm_obj.score_record['Fscore']['average_value'],
    # both3_im_obj.score_record['AUC']['average_value'], both3_im_obj.score_record['Fscore']['average_value'],
    # both3_both_obj.score_record['AUC']['average_value'], both3_both_obj.score_record['Fscore']['average_value'])


    # table_2 += '\\hline\n'

    # table_2 += '\\end{tabular}\n'
    # table_2 += '\\end{table}\n'

    # table_2 += '\n\n\n'

    # total_record += table_2

# with open('result_table.txt','w') as record:
#     # record.write(table_1)
#     record.write(total_record)



























# def all_metrix():
#     # --------------------------------- 全部指标 ----------------------------------

        

#     dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']

#     data_range = 5
#     record_index = 1

#     train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
#     early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
#     # early_stop_type_list = [ '20000', '15000', '10000', '8000']
#     # early_stop_type_list = [ '5000', '2000']
#     # early stop 效果不太明显， 结果不太好
#     test_infor_method_list = [ 'normal', 'bm', 'im', 'both']
#     # test_infor_method_list = [ 'normal', 'bm', 'im', 'both']

#     ref_num_type_list = ['num']
#     ref_times_list = ['10']
#     boundary_type_list = ['half']

#     # transform_list = ['concat', 'minus']
#     # mirror_type_list = ['Mirror', 'notMirror']
#     # early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
#     # early stop 效果不太明显， 结果不太好

#     score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]
#     for dataset in dataset_list:
#         with open('dataset_{0}_result.txt'.format(dataset), 'w') as f:
#             command_list = []
#             first_line = 'method_name\t'
#             for score_type in score_type_list:
#                 cur_part = ''
                
#                 cur_part += '{0}_average_value\t'.format(score_type)
#                 cur_part += '{0}_max_value\t'.format(score_type)
#                 cur_part += '{0}_min_value\t'.format(score_type)

#                 for index in range(1, 6):
#                     cur_part += score_type + 'all_value_{0}\t'.format(index)
                

#                 first_line += cur_part
#             f.write(first_line + '\n')

#             for sample_method in train_infor_method_list:
#                 for early_stop_type in early_stop_type_list:
#                     for test_infor_method in test_infor_method_list:
#                         for ref_num_type in ref_num_type_list:
#                             for ref_times in ref_times_list:
#                                 for boundary_type in boundary_type_list:
#                                     cur_command_list = []
#                                     train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
#                                     test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
#                                     cur_dataset_path = './test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
#                                     cur_file_name_prefix = dataset
#                                     cur_method = '{0}_{1}'.format(train_method, test_method)
                                    
#                                     print(cur_dataset_path)
#                                     cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
#                                     cur_output_str = cur_obj.get_print_str()
#                                     f.write(cur_output_str + '\n')
#                                     print('end')
#             f.write('\n\n\n')



# def main_metrix():
#     # --------------------------------- only average min max metrix ----------------------------------

        

#     dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']

#     data_range = 5
#     record_index = 1

#     train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
#     early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
#     # early_stop_type_list = [ '20000', '15000', '10000', '8000']
#     # early_stop_type_list = [ '5000', '2000']
#     # early stop 效果不太明显， 结果不太好
#     test_infor_method_list = [ 'normal', 'bm', 'im', 'both']
#     # test_infor_method_list = [ 'normal', 'bm', 'im', 'both']

#     ref_num_type_list = ['num']
#     ref_times_list = ['10']
#     boundary_type_list = ['half']

#     # transform_list = ['concat', 'minus']
#     # mirror_type_list = ['Mirror', 'notMirror']
#     # early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
#     # early stop 效果不太明显， 结果不太好

#     score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]
#     for dataset in dataset_list:
#         with open('dataset_{0}_result.txt'.format(dataset), 'w') as f:
#             command_list = []
#             first_line = 'method_name\t'
#             for score_type in score_type_list:
#                 cur_part = ''
                
#                 cur_part += '{0}_average_value\t'.format(score_type)
#                 cur_part += '{0}_max_value\t'.format(score_type)
#                 cur_part += '{0}_min_value'.format(score_type)
#                 if score_type != 'AUC':
#                     cur_part += '\t'

#                 first_line += cur_part
#             f.write(first_line + '\n')

#             for sample_method in train_infor_method_list:
#                 for early_stop_type in early_stop_type_list:
#                     for test_infor_method in test_infor_method_list:
#                         for ref_num_type in ref_num_type_list:
#                             for ref_times in ref_times_list:
#                                 for boundary_type in boundary_type_list:
#                                     cur_command_list = []
#                                     train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
#                                     test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
#                                     cur_dataset_path = './test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
#                                     cur_file_name_prefix = dataset
#                                     cur_method = '{0}_{1}'.format(train_method, test_method)
                                    
#                                     print(cur_dataset_path)
#                                     cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
#                                     cur_output_str = cur_obj.get_summary_print_str()
#                                     f.write(cur_output_str + '\n')
#                                     print('end')
#             f.write('\n\n\n')


# all_metrix()
        
        # cur_dataset_path = './test_abalone19/result_MLP_3_40000_normal/record_1/'
        # cur_file_name_prefix = 'abalone19'
        # cur_method = 'MLP_3_40000_normal'
        # print(cur_dataset_path)
        # cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
        # cur_output_str = cur_obj.get_print_str()
        # f.write(cur_output_str + '\n')
        # print('end')


#                                         cur_command_list.append('mkdir -p ./test_{0}/model_{2}/record_{1}/\n'.format(dataset, record_index, train_method)) 
#                                         cur_command_list.append('mkdir -p ./test_{0}/result_{1}_{2}/record_{3}/\n'.format(dataset, train_method, test_method, record_index))
#                                         for dataset_index in range(1, 6):
#                                             # cur_command_list.append('python3 ./classifier_MLP/train_MLP.py dataset_name={0} dataset_index={1} record_index=1 device_id={2} train_method={3}\n'.format(dataset, dataset_index, device_id, train_method))
#                                             cur_command_list.append('python3 ./classifier_MLP/test.py dataset_name={0} dataset_index={1} record_index=1 train_method={2} test_method={3} device_id={4}\n'.format(dataset, dataset_index, train_method, test_method, device_id))
#                                         cur_command_list.append('\n\n\n')
#                                         command_list.append(cur_command_list)
                

#             fsh.write('#!/bin/bash\n')
#             fsh.write('set -e\n\n\n')
#             for cur_command_list in command_list:
#                 for line in cur_command_list:
#                     fsh.write(line)

    
# with open('all_dataset_normal_result.txt', 'w') as f:
#     for dataset in dataset_list:
#         for model_type in model_type_list:
#             for transform_method in transform_list:
#                 for mirror_type in mirror_type_list:
#                     train_method = '{0}_normal'.format(model_type)
#                     test_method = 'normal'
#                     cur_dataset_path = '../test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
#                     cur_file_name_prefix = dataset
#                     cur_method = '{0}_{1}'.format(train_method, test_method)
#                     print(cur_dataset_path)
#                     cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
#                     cur_output_str = cur_obj.get_print_str()
#                     f.write(cur_output_str + '\n')
#                     print('end')
#         f.write('\n\n\n')
























# -------------------------------------global parameters---------------------------------
# table_1 = '\\begin{table}[H]\n'
# table_1 += '\\centering\n'
# table_1 += '\\caption{the performance of different varienties of ijcai method}\n'
# table_1 += '\\label{tab:ChangingTrainData33}\n'
# table_1 += '\\begin{tabular}{|p{0.1\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|}\n'
# table_1 += '\\hline \multirow{2}{*}{Method} & \multicolumn{4}{|c|}{Original test} & \multicolumn{4}{|c|}{Border Majority} & \multicolumn{4}{|c|}{Informative Minority} & \multicolumn{4}{|c|}{Both of them}\\\\\n'
# table_1 += '\\cline{2-17} & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall \\\\\n'
# table_1 += '\\hline IJCAI & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_normal_normal_record.total_average_value_aauc, ijcai_normal_normal_record.total_average_value_af, ijcai_normal_normal_record.total_average_value_ap, ijcai_normal_normal_record.total_average_value_ar, ijcai_normal_bm_record.total_average_value_aauc, ijcai_normal_bm_record.total_average_value_af, ijcai_normal_bm_record.total_average_value_ap, ijcai_normal_bm_record.total_average_value_ar, ijcai_normal_im_record.total_average_value_aauc, ijcai_normal_im_record.total_average_value_af, ijcai_normal_im_record.total_average_value_ap, ijcai_normal_im_record.total_average_value_ar, ijcai_normal_both_record.total_average_value_aauc, ijcai_normal_both_record.total_average_value_af, ijcai_normal_both_record.total_average_value_ap, ijcai_normal_both_record.total_average_value_ar)
# table_1 += '\\hline Training with BM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_bm_normal_record.total_average_value_aauc, ijcai_bm_normal_record.total_average_value_af, ijcai_bm_normal_record.total_average_value_ap, ijcai_bm_normal_record.total_average_value_ar, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_ap, ijcai_bm_bm_record.total_average_value_ar, ijcai_bm_im_record.total_average_value_aauc, ijcai_bm_im_record.total_average_value_af, ijcai_bm_im_record.total_average_value_ap, ijcai_bm_im_record.total_average_value_ar, ijcai_bm_both_record.total_average_value_aauc, ijcai_bm_both_record.total_average_value_af, ijcai_bm_both_record.total_average_value_ap, ijcai_bm_both_record.total_average_value_ar)
# table_1 += '\\hline Training with IM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_im_normal_record.total_average_value_aauc, ijcai_im_normal_record.total_average_value_af, ijcai_im_normal_record.total_average_value_ap, ijcai_im_normal_record.total_average_value_ar, ijcai_im_bm_record.total_average_value_aauc, ijcai_im_bm_record.total_average_value_af, ijcai_im_bm_record.total_average_value_ap, ijcai_im_bm_record.total_average_value_ar, ijcai_im_im_record.total_average_value_aauc, ijcai_im_im_record.total_average_value_af, ijcai_im_im_record.total_average_value_ap, ijcai_im_im_record.total_average_value_ar, ijcai_im_both_record.total_average_value_aauc, ijcai_im_both_record.total_average_value_af, ijcai_im_both_record.total_average_value_ap, ijcai_im_both_record.total_average_value_ar)
# table_1 += '\\hline Training wiht BM and IM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_both_normal_record.total_average_value_aauc, ijcai_both_normal_record.total_average_value_af, ijcai_both_normal_record.total_average_value_ap, ijcai_both_normal_record.total_average_value_ar, ijcai_both_bm_record.total_average_value_aauc, ijcai_both_bm_record.total_average_value_af, ijcai_both_bm_record.total_average_value_ap, ijcai_both_bm_record.total_average_value_ar, ijcai_both_im_record.total_average_value_aauc, ijcai_both_im_record.total_average_value_af, ijcai_both_im_record.total_average_value_ap, ijcai_both_im_record.total_average_value_ar, ijcai_both_both_record.total_average_value_aauc, ijcai_both_both_record.total_average_value_af, ijcai_both_both_record.total_average_value_ap, ijcai_both_both_record.total_average_value_ar)
# table_1 += '\\hline\n'

# table_1 += '\\end{tabular}\n'
# table_1 += '\\end{table}\n'




# table_2 = '\\begin{table}[H]\n'
# table_2 += '\\centering\n'
# table_2 += '\\caption{the performance of different varienties of ijcai method (AUC and F1}\n'
# table_2 += '\\label{tab:ChangingTrainData33}\n'
# table_2 += '\\begin{tabular}{|p{0.1\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|}\n'
# table_2 += '\\hline \multirow{2}{*}{Method} & \multicolumn{2}{|c|}{Original test} & \multicolumn{2}{|c|}{Border Majority} & \multicolumn{2}{|c|}{Informative Minority} & \multicolumn{2}{|c|}{Both of them}\\\\\n'
# table_2 += '\\cline{2-9} & Auc & F1  & Auc & F1 & Auc & F1 & Auc & F1  \\\\\n'
# table_2 += '\\hline IJCAI & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_normal_normal_record.total_average_value_aauc, ijcai_normal_normal_record.total_average_value_af, ijcai_normal_bm_record.total_average_value_aauc, ijcai_normal_bm_record.total_average_value_af, ijcai_normal_im_record.total_average_value_aauc, ijcai_normal_im_record.total_average_value_af, ijcai_normal_both_record
#     .total_average_value_aauc, ijcai_normal_both_record
# .total_average_value_af)
# table_2 += '\\hline Training with BM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_bm_normal_record.total_average_value_aauc, ijcai_bm_normal_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_bm_im_record.total_average_value_aauc,ijcai_bm_im_record.total_average_value_af, ijcai_bm_both_record.total_average_value_aauc, ijcai_bm_both_record.total_average_value_af)
# table_2 += '\\hline Training with IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_im_normal_record.total_average_value_aauc, ijcai_im_normal_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_im_im_record.total_average_value_aauc, ijcai_im_im_record.total_average_value_af, ijcai_im_both_record.total_average_value_aauc, ijcai_im_both_record.total_average_value_af)
# table_2 += '\\hline Training wiht BM and IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_both_normal_record.total_average_value_aauc, ijcai_both_normal_record.total_average_value_af, ijcai_both_bm_record.total_average_value_aauc, ijcai_both_bm_record.total_average_value_af, ijcai_both_im_record.total_average_value_aauc, ijcai_both_im_record.total_average_value_af, ijcai_both_both_record.total_average_value_aauc, ijcai_both_both_record.total_average_value_af)
# table_2 += '\\hline\n'

# table_2 += '\\end{tabular}\n'
# table_2 += '\\end{table}\n'

# record = open('result_table.txt','w')
# record.write(table_1)
# record.write(table_2)


























# dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']

# data_range = 5
# record_index = 1

# train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
# early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# # early_stop_type_list = [ '20000', '15000', '10000', '8000']
# # early_stop_type_list = [ '5000', '2000']
# # early stop 效果不太明显， 结果不太好
# test_infor_method_list = [ 'normal', 'bm', 'im', 'both']
# # test_infor_method_list = [ 'normal', 'bm', 'im', 'both']

# ref_num_type_list = ['num']
# ref_times_list = ['10']
# boundary_type_list = ['half']

# # transform_list = ['concat', 'minus']
# # mirror_type_list = ['Mirror', 'notMirror']
# # early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# # early stop 效果不太明显， 结果不太好

# score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]
# with open('all_dataset_normal_result.txt', 'w') as f:
#     command_list = []
#     first_line = 'method_name\t'
#     for score_type in score_type_list:
#         cur_part = ''
#         for index in range(1, 6):
#             cur_part += score_type + 'all_value_{0}\t'.format(index)
#         cur_part += 'average_value\t'
#         cur_part += 'max_value\t'
#         if score_type == 'AUC':
#             cur_part += 'min_value'
#         else:
#             cur_part += 'min_value\t'
#         first_line += cur_part
#     f.write(first_line + '\n')
#     for dataset in dataset_list:
#         for sample_method in train_infor_method_list:
#             for early_stop_type in early_stop_type_list:
#                 for test_infor_method in test_infor_method_list:
#                     for ref_num_type in ref_num_type_list:
#                         for ref_times in ref_times_list:
#                             for boundary_type in boundary_type_list:
#                                 cur_command_list = []
#                                 train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
#                                 test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
#                                 cur_dataset_path = './test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
#                                 cur_file_name_prefix = dataset
#                                 cur_method = '{0}_{1}'.format(train_method, test_method)
                                
#                                 print(cur_dataset_path)
#                                 cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
#                                 cur_output_str = cur_obj.get_print_str()
#                                 f.write(cur_output_str + '\n')
#                                 print('end')
#         f.write('\n\n\n')



#                                         cur_command_list.append('mkdir -p ./test_{0}/model_{2}/record_{1}/\n'.format(dataset, record_index, train_method)) 
#                                         cur_command_list.append('mkdir -p ./test_{0}/result_{1}_{2}/record_{3}/\n'.format(dataset, train_method, test_method, record_index))
#                                         for dataset_index in range(1, 6):
#                                             # cur_command_list.append('python3 ./classifier_MLP/train_MLP.py dataset_name={0} dataset_index={1} record_index=1 device_id={2} train_method={3}\n'.format(dataset, dataset_index, device_id, train_method))
#                                             cur_command_list.append('python3 ./classifier_MLP/test.py dataset_name={0} dataset_index={1} record_index=1 train_method={2} test_method={3} device_id={4}\n'.format(dataset, dataset_index, train_method, test_method, device_id))
#                                         cur_command_list.append('\n\n\n')
#                                         command_list.append(cur_command_list)
                

#             fsh.write('#!/bin/bash\n')
#             fsh.write('set -e\n\n\n')
#             for cur_command_list in command_list:
#                 for line in cur_command_list:
#                     fsh.write(line)

    
# with open('all_dataset_normal_result.txt', 'w') as f:
#     for dataset in dataset_list:
#         for model_type in model_type_list:
#             for transform_method in transform_list:
#                 for mirror_type in mirror_type_list:
#                     train_method = '{0}_normal'.format(model_type)
#                     test_method = 'normal'
#                     cur_dataset_path = '../test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
#                     cur_file_name_prefix = dataset
#                     cur_method = '{0}_{1}'.format(train_method, test_method)
#                     print(cur_dataset_path)
#                     cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
#                     cur_output_str = cur_obj.get_print_str()
#                     f.write(cur_output_str + '\n')
#                     print('end')
#         f.write('\n\n\n')
























# -------------------------------------global parameters---------------------------------
# table_1 = '\\begin{table}[H]\n'
# table_1 += '\\centering\n'
# table_1 += '\\caption{the performance of different varienties of ijcai method}\n'
# table_1 += '\\label{tab:ChangingTrainData33}\n'
# table_1 += '\\begin{tabular}{|p{0.1\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|}\n'
# table_1 += '\\hline \multirow{2}{*}{Method} & \multicolumn{4}{|c|}{Original test} & \multicolumn{4}{|c|}{Border Majority} & \multicolumn{4}{|c|}{Informative Minority} & \multicolumn{4}{|c|}{Both of them}\\\\\n'
# table_1 += '\\cline{2-17} & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall \\\\\n'
# table_1 += '\\hline IJCAI & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_normal_normal_record.total_average_value_aauc, ijcai_normal_normal_record.total_average_value_af, ijcai_normal_normal_record.total_average_value_ap, ijcai_normal_normal_record.total_average_value_ar, ijcai_normal_bm_record.total_average_value_aauc, ijcai_normal_bm_record.total_average_value_af, ijcai_normal_bm_record.total_average_value_ap, ijcai_normal_bm_record.total_average_value_ar, ijcai_normal_im_record.total_average_value_aauc, ijcai_normal_im_record.total_average_value_af, ijcai_normal_im_record.total_average_value_ap, ijcai_normal_im_record.total_average_value_ar, ijcai_normal_both_record.total_average_value_aauc, ijcai_normal_both_record.total_average_value_af, ijcai_normal_both_record.total_average_value_ap, ijcai_normal_both_record.total_average_value_ar)
# table_1 += '\\hline Training with BM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_bm_normal_record.total_average_value_aauc, ijcai_bm_normal_record.total_average_value_af, ijcai_bm_normal_record.total_average_value_ap, ijcai_bm_normal_record.total_average_value_ar, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_ap, ijcai_bm_bm_record.total_average_value_ar, ijcai_bm_im_record.total_average_value_aauc, ijcai_bm_im_record.total_average_value_af, ijcai_bm_im_record.total_average_value_ap, ijcai_bm_im_record.total_average_value_ar, ijcai_bm_both_record.total_average_value_aauc, ijcai_bm_both_record.total_average_value_af, ijcai_bm_both_record.total_average_value_ap, ijcai_bm_both_record.total_average_value_ar)
# table_1 += '\\hline Training with IM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_im_normal_record.total_average_value_aauc, ijcai_im_normal_record.total_average_value_af, ijcai_im_normal_record.total_average_value_ap, ijcai_im_normal_record.total_average_value_ar, ijcai_im_bm_record.total_average_value_aauc, ijcai_im_bm_record.total_average_value_af, ijcai_im_bm_record.total_average_value_ap, ijcai_im_bm_record.total_average_value_ar, ijcai_im_im_record.total_average_value_aauc, ijcai_im_im_record.total_average_value_af, ijcai_im_im_record.total_average_value_ap, ijcai_im_im_record.total_average_value_ar, ijcai_im_both_record.total_average_value_aauc, ijcai_im_both_record.total_average_value_af, ijcai_im_both_record.total_average_value_ap, ijcai_im_both_record.total_average_value_ar)
# table_1 += '\\hline Training wiht BM and IM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_both_normal_record.total_average_value_aauc, ijcai_both_normal_record.total_average_value_af, ijcai_both_normal_record.total_average_value_ap, ijcai_both_normal_record.total_average_value_ar, ijcai_both_bm_record.total_average_value_aauc, ijcai_both_bm_record.total_average_value_af, ijcai_both_bm_record.total_average_value_ap, ijcai_both_bm_record.total_average_value_ar, ijcai_both_im_record.total_average_value_aauc, ijcai_both_im_record.total_average_value_af, ijcai_both_im_record.total_average_value_ap, ijcai_both_im_record.total_average_value_ar, ijcai_both_both_record.total_average_value_aauc, ijcai_both_both_record.total_average_value_af, ijcai_both_both_record.total_average_value_ap, ijcai_both_both_record.total_average_value_ar)
# table_1 += '\\hline\n'

# table_1 += '\\end{tabular}\n'
# table_1 += '\\end{table}\n'




# table_2 = '\\begin{table}[H]\n'
# table_2 += '\\centering\n'
# table_2 += '\\caption{the performance of different varienties of ijcai method (AUC and F1}\n'
# table_2 += '\\label{tab:ChangingTrainData33}\n'
# table_2 += '\\begin{tabular}{|p{0.1\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|}\n'
# table_2 += '\\hline \multirow{2}{*}{Method} & \multicolumn{2}{|c|}{Original test} & \multicolumn{2}{|c|}{Border Majority} & \multicolumn{2}{|c|}{Informative Minority} & \multicolumn{2}{|c|}{Both of them}\\\\\n'
# table_2 += '\\cline{2-9} & Auc & F1  & Auc & F1 & Auc & F1 & Auc & F1  \\\\\n'
# table_2 += '\\hline IJCAI & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_normal_normal_record.total_average_value_aauc, ijcai_normal_normal_record.total_average_value_af, ijcai_normal_bm_record.total_average_value_aauc, ijcai_normal_bm_record.total_average_value_af, ijcai_normal_im_record.total_average_value_aauc, ijcai_normal_im_record.total_average_value_af, ijcai_normal_both_record
#     .total_average_value_aauc, ijcai_normal_both_record
# .total_average_value_af)
# table_2 += '\\hline Training with BM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_bm_normal_record.total_average_value_aauc, ijcai_bm_normal_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_bm_im_record.total_average_value_aauc,ijcai_bm_im_record.total_average_value_af, ijcai_bm_both_record.total_average_value_aauc, ijcai_bm_both_record.total_average_value_af)
# table_2 += '\\hline Training with IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_im_normal_record.total_average_value_aauc, ijcai_im_normal_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_im_im_record.total_average_value_aauc, ijcai_im_im_record.total_average_value_af, ijcai_im_both_record.total_average_value_aauc, ijcai_im_both_record.total_average_value_af)
# table_2 += '\\hline Training wiht BM and IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_both_normal_record.total_average_value_aauc, ijcai_both_normal_record.total_average_value_af, ijcai_both_bm_record.total_average_value_aauc, ijcai_both_bm_record.total_average_value_af, ijcai_both_im_record.total_average_value_aauc, ijcai_both_im_record.total_average_value_af, ijcai_both_both_record.total_average_value_aauc, ijcai_both_both_record.total_average_value_af)
# table_2 += '\\hline\n'

# table_2 += '\\end{tabular}\n'
# table_2 += '\\end{table}\n'

# record = open('result_table.txt','w')
# record.write(table_1)
# record.write(table_2)