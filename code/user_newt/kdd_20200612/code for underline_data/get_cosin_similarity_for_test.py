import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

test_merge_file = '..\\test_data\\test_merge_phase-%d.csv'
user_path = os.path.expanduser('~')
train_path = os.path.join(user_path, 'kdd\\data\\underexpose_train')
item_feat_file = os.path.join(train_path, 'underexpose_item_feat.csv')

user_set = set()
item_set = set()
txt_list = []
default_txt_list = [-1.0] * 128  #对于没有txt_vec和img_vec的item，txt_vec和img_vec设置为-1
img_list = []
item_txt = defaultdict(list)
item_img = defaultdict(list)
users_new=set()
recall_num=500

with open(item_feat_file, 'r') as fin:
    for line in fin:
        line = line.strip().split(',[')
        item_id = int(line[0])
        item_set.add(item_id)
        txt_vec = [float(x) for x in line[1].strip(']').split(',')]
        img_vec=[float(x) for x in line[2].strip(']').split(',')]
        item_txt[item_id] = txt_vec
        item_img[item_id] = img_vec


for phase in range(7, 10):
    print('phase:',phase)
    
    with open('..\\test_data\\dict_test_user_items_phase_%d.txt' % phase, 'r') as fout:
        temp_str = fout.read()
        dict_user_items = eval(temp_str)
        print('len(dict_user_items): ', len(dict_user_items))

    df_test = pd.read_csv(test_merge_file % phase,
                           header=None,
                           names=['user_id', 'item_id', 'itemcf_rank', 'itemcf_score', 'label', 'usercf_rank',
                                  'usercf_score', 'w2v_rank', 'w2v_score', 'item_heat', 'user_heat'])
    print('\ndf_test: ', df_test.shape)
    print(df_test.head())

    user_set=set(df_test['user_id'].values)
    print('len(user_set): ',len(user_set))

    user_itemlist_df=df_test.groupby('user_id')['item_id'].agg(list)

    user_lists=[]
    item_lists=[]
    txt_sim_lists=[]
    img_sim_lists=[]

    for user in tqdm(user_set):
        if user not in dict_user_items: # 用户没有购买历史
            users_new.add(user)
            continue

        item_list_0 = dict_user_items[user]
        item_list_1 = user_itemlist_df[user]
        txt_list_0 = []
        txt_list_1 = []
        img_list_0 = []
        img_list_1 = []

        for item_0 in item_list_0:
            if item_0 in item_set:
                txt_list_0.append(item_txt[item_0])
                img_list_0.append(item_img[item_0])
            else:
                txt_list_0.append(default_txt_list)
                img_list_0.append(default_txt_list)


        for item_1 in item_list_1:
            user_lists.append(user)
            item_lists.append(item_1)
            if item_1 in item_set:
                txt_list_1.append(item_txt[item_1])
                img_list_1.append(item_img[item_1])
            else:
                txt_list_1.append(default_txt_list)
                img_list_1.append(default_txt_list)

        txt_array_0 = np.array(txt_list_0)
        img_array_0 = np.array(img_list_0)
        # print(txt_array_0.shape)
        txt_array_1 = np.array(txt_list_1)
        img_array_1 = np.array(img_list_1)
        # print(txt_array_1.shape)

        sim_text_mat = cosine_similarity(txt_array_1, txt_array_0)
        value_to_fill = sim_text_mat.mean(axis=1)
        index_list_0, index_list_1 = np.where(sim_text_mat < -1e-9)
        sim_text_mat[index_list_0, index_list_1] = value_to_fill[index_list_0]
        sim_text_mat = sim_text_mat.mean(axis=1)
        txt_sim_lists+=list(sim_text_mat)

        sim_img_mat = cosine_similarity(img_array_1, img_array_0)
        value_to_fill = sim_img_mat.mean(axis=1)
        index_list_0, index_list_1 = np.where(sim_img_mat < -1e-9)
        sim_img_mat[index_list_0, index_list_1] = value_to_fill[index_list_0]
        sim_img_mat = sim_img_mat.mean(axis=1)
        img_sim_lists += list(sim_img_mat)

    temp_dict=dict()
    temp_dict['user_id'] = user_lists
    temp_dict['item_id'] = item_lists
    temp_dict['txt_sim'] = txt_sim_lists
    temp_dict['img_sim'] = img_sim_lists

    df_new=pd.DataFrame(data=temp_dict)
    print('\ndf_new: ',df_new.shape)
    print(df_new.head())
    print('len(users_new): ',len(users_new))

    # df_test=df_test[~df_test['user_id'].isin(users_new)]
    df=pd.merge(df_new, df_test, how='left', on=['user_id','item_id'])
    print('df: ',df.shape)
    print(df.head())
    print(df.columns.values)
    df.to_csv('..\\test_data\\test_feat_phase-%d.csv' % phase, index=False, header=False)







