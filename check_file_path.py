
# # -*- coding:utf-8 -*-
import os




# ---------------------  检查 test -----------------------------------
import sys
import os
dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']

data_range = 5
record_index = 1

train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# early_stop_type_list = [ '20000', '15000', '10000', '8000']
# early_stop_type_list = [ '5000', '2000']
# early stop 效果不太明显， 结果不太好
# test_infor_method_list = ['normal']
test_infor_method_list = [ 'normal', 'bm', 'im', 'both']

ref_num_type_list = ['num']
ref_times_list = ['10']
boundary_type_list = ['half']


# for file_index in dataset_dict:
    # dataset_list = dataset_dict[file_index]
device_id = 0

command_list = []
for dataset in dataset_list:
    for sample_method in train_infor_method_list:
        for early_stop_type in early_stop_type_list:
            for test_infor_method in test_infor_method_list:
                for ref_num_type in ref_num_type_list:
                    for ref_times in ref_times_list:
                        for boundary_type in boundary_type_list:
                            cur_command_list = []
                            train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
                            test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)

                            cur_path = './test_{0}/result_{1}_{2}/'.format(dataset, train_method, test_method)
                            if os.path.exists(cur_path):
                                cur_command_list.append('rm -rf ./test_{0}/model_{2}/\n'.format(dataset, record_index, train_method)) 

                            cur_path = './test_{0}/model_{1}/'.format(dataset, train_method)
                            if os.path.exists(cur_path):
                                cur_command_list.append('rm -rf ./test_{0}/result_{1}_{2}\n'.format(dataset, train_method, test_method))
                                
                            if len(cur_command_list) != 0:
                                cur_command_list.append('\n')
                                command_list.append(cur_command_list)

        
with open('del_file.sh', 'w') as fsh:
    fsh.write('#!/bin/bash\n')
    fsh.write('set -e\n\n\n')
    for cur_command_list in command_list:
        for line in cur_command_list:
            fsh.write(line)






# # print(os.path.exists('/Users/taihan/Documents/test_ijcai_extention/test_glass5/model_MLP_im_8000/record_1/MLP_im_8000_6'))


# dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']

# data_range = 5
# record_index = 1

# train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
# early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# # early_stop_type_list = [ '20000', '15000', '10000', '8000']
# # early_stop_type_list = [ '5000', '2000']
# # early stop 效果不太明显， 结果不太好
# # test_infor_method_list = ['normal']
# test_infor_method_list = [ 'normal' 'bm', 'im', 'both']

# ref_num_type_list = ['num']
# ref_times_list = ['10']
# boundary_type_list = ['half']

# # transform_list = ['concat', 'minus']
# # mirror_type_list = ['Mirror', 'notMirror']
# # early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# # early stop 效果不太明显， 结果不太好


# loss_list = []
# cnt = 0
# for dataset in dataset_list:
#     for sample_method in train_infor_method_list:
#         for early_stop_type in early_stop_type_list:
#             train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
#             for dataset_index in range(1, 6):
#                 cur_path = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset, train_method, record_index, dataset_index)
#                 if os.path.exists(cur_path):
#                     pass
#                 else:
#                     cnt += 1
#                     print(cur_path)
#                     print(dataset, train_method, dataset_index)
#                     print('\n\n\n')
#                     loss_list.append([dataset, train_method, dataset_index])

# print(cnt)
# print(loss_list)

            
            
            
            
            
            
#             for test_infor_method in test_infor_method_list:
#                 for ref_num_type in ref_num_type_list:
#                     for ref_times in ref_times_list:
#                         for boundary_type in boundary_type_list:
#                             cur_command_list = []
#                             train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
#                             test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
#                             cur_dataset_path = './test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
#                             cur_file_name_prefix = dataset
#                             cur_method = '{0}_{1}'.format(train_method, test_method)
                            
#                             print(cur_dataset_path)
#                             cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
#                             cur_output_str = cur_obj.get_print_str()
#                             f.write(cur_output_str + '\n')
#                             print('end')


# # with open('all_dataset_normal_result.txt', 'w') as f:
# #         dataset_list = dataset_dict[file_index]
# #         device_id = device_id_map[file_index]
# #         with open('train_mlp_{0}.sh'.format(file_index), 'w') as fsh:
# #             command_list = []
# #             for dataset in dataset_list:
# #                 for sample_method in train_infor_method_list:
# #                     for early_stop_type in early_stop_type_list:
# #                         for test_infor_method in test_infor_method_list:
# #                             for ref_num_type in ref_num_type_list:
# #                                 for ref_times in ref_times_list:
# #                                     for boundary_type in boundary_type_list:
# #                                         cur_command_list = []
# #                                         train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
# #                                         test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
# #                                         cur_dataset_path = './test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
# #                                         cur_file_name_prefix = dataset
# #                                         cur_method = '{0}_{1}'.format(train_method, test_method)
                                        
# #                                         print(cur_dataset_path)
# #                                         cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
# #                                         cur_output_str = cur_obj.get_print_str()
# #                                         f.write(cur_output_str + '\n')
# #                                         print('end')
# #                 f.write('\n\n\n')
