from collections import defaultdict
import pandas as pd
import math
import os
import numpy as np
from tqdm import tqdm

def usercf(df_):
    df = df_.copy()
    item_users = df.groupby(['item_id'])['user_id'].agg(list).reset_index()
    item_users_dict = dict(zip(item_users['item_id'], item_users['user_id']))
    C = dict()
    N = defaultdict(lambda: 0)
    for i, users in item_users_dict.items():
        for u in users:
            N[u] += 1
            if u not in C:
                C[u] = {}
            for v in users:
                if u == v:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
                C[u][v] += 1 / math.log(1 + len(users))

    W = defaultdict(dict)
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W


def recall_usercf(sim_user_corr, user_items_dict, user_id, top_k, item_num):
    '''

    :param sim_user_corr: 用户相似度矩阵
    :param user_items_dict: 用户-点击过的商品列表
    :param user_id:
    :param top_k:
    :param item_num:
    :return:
    '''
    item_score_dict = defaultdict(lambda: 0)
    if user_id not in user_items_dict:
        return []
    interacted_items = user_items_dict[user_id]
    users_sim = sim_user_corr[user_id]
    for related_user, sim in sorted(users_sim.items(), key=lambda x: x[1], reverse=True)[:top_k]:  # related_user相似用户, sim相似度
        for item in user_items_dict[related_user]:  #related_user的item点击历史
            if item in interacted_items:
                continue
            item_score_dict[item] += sim
    return sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)[:item_num]

start_phase=7
now_phase = 9
user_path = os.path.expanduser('~')
train_path = os.path.join(user_path, r'kdd\data\underexpose_train')
test_path = os.path.join(user_path, r'kdd\data\underexpose_test')
click = pd.DataFrame()
item_deg = defaultdict(lambda: 0)
interacted_user_items_dict = dict()
recall_num = 500

num_user_test = 0
num_user_val=0

for c in range(start_phase, now_phase + 1):

    print('phase:', c)

    user_lists = []
    item_lists = []
    item_sim_lists = []
    label_lists = []
    item_rank_lists = []

    click_train = pd.read_csv(train_path + '\\underexpose_train_click-{}.csv'.format(c), header=None, nrows=None,
                              names=['user_id', 'item_id', 'time'],
                              dtype={'user_id': np.int, 'item_id': np.int, 'time': np.float32})
    click_test = pd.read_csv(test_path + '\\underexpose_test_click-{}.csv'.format(c), header=None, nrows=None,
                             names=['user_id', 'item_id', 'time'],
                             dtype={'user_id': np.int, 'item_id': np.int, 'time': np.float32})

    user_test = set(click_test['user_id'])  # 每阶段线下测试集用户集合
    print('len(user_test):', len(user_test))
    num_user_test+=len(user_test)

    click_train_test = click_train.append(click_test)
    click = click.append(click_train_test)  # 当前阶段以及之前阶段click数据
    # click=click_train_test                        #  当前阶段click数据

    click = click.sort_values('time')  # 时间排序
    click = click.drop_duplicates(['user_id', 'item_id', 'time'], keep='last')  # 去重

    underline_train_val=click


    user_itemlist = click.groupby('user_id')['item_id'].agg(list).reset_index()
    user_items_dict = dict(zip(user_itemlist['user_id'], user_itemlist['item_id']))
    #
    item_userlist = underline_train_val.groupby('item_id')[['user_id']].agg(list).reset_index()
    dict_item_userlist = dict(zip(item_userlist['item_id'], item_userlist['user_id']))
    #
    top500_click = list(underline_train_val['item_id'].value_counts().index[:500].values)  # 最热商品
    #
    # item_sim_corr = itemcf(underline_train, user_items_dict, use_iif=False)

    user_sim_corr = usercf(click)

    for user in tqdm(user_test):

        user_list = []
        item_list = []
        item_sim_list = []  # item分数
        label_list = []     # 标签
        item_rank_list=[]   # item 排名

        # rank_item_0 = recall_itemcf(item_sim_corr, user_items_dict, user, 500, recall_num)
        rank_item_1 = recall_usercf(user_sim_corr, user_items_dict, user, 500, recall_num)
        rank=1

        # if rank_item_0:
        #     for item in rank_item_0:
        #         if item[0] not in item_list:
        #             item_list.append(item[0])
        #             item_sim_list.append(item[1])
        #             item_rank_list.append(rank)
        #             label_list.append(int(item[0] == target_item))
        #             rank+=1

        if rank_item_1:
            for item in rank_item_1:
                if item[0] not in item_list:
                    user_list.append(user)
                    item_list.append(item[0])
                    item_sim_list.append(item[1])
                    item_rank_list.append(rank)
                    label_list.append(0)   # 线上测试没有label,只是为了和线下测试集dataframe结构一致
                    rank+=1

        # if len(item_list) < recall_num:
        #     for i, item in enumerate(top500_click):
        #         if item not in item_list:
        #             item_list.append(item)
        #             item_sim_list.append(-1.0*(i+1))
        #             label_list.append(int(item == target_item))
        #             item_rank_list.append(rank)
        #             rank+=1
        #         if len(item_list) == recall_num:
        #             break

        user_lists += user_list
        item_lists += item_list
        item_sim_lists += item_sim_list
        label_lists += label_list
        item_rank_lists+=item_rank_list

    temp_dict = dict()
    temp_dict['user_id'] = user_lists
    temp_dict['item_id'] = item_lists
    temp_dict['usercf_rank']=item_rank_lists
    temp_dict['usercf_score'] = item_sim_lists
    temp_dict['label']=label_lists


    df = pd.DataFrame(data=temp_dict)
    print(df.shape)
    assert df.shape[1] == 5
    print(df.head())
    df.to_csv('..\\online_data\\online_usercf_phase-%d.csv' %c, index=False, header=False)


