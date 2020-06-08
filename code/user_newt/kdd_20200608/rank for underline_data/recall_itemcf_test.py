from collections import defaultdict
import pandas as pd
import math
import os
import numpy as np
from tqdm import tqdm


def itemcf(df_, user_items_dict, use_iif=False):
    df = df_.copy()
    user_time_ = df.groupby('user_id')['time'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))

    sim_item = {}
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in user_items_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                t1 = user_time_dict[user][loc1]  # 点击时间提取
                t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)
                if not use_iif:
                    if loc1 - loc2 > 0:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8 ** (loc1 - loc2 - 1)) * (
                                1 - (t1 - t2) * 10000) / math.log(1 + len(items))  # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8 ** (loc2 - loc1 - 1)) * (
                                1 - (t2 - t1) * 10000) / math.log(1 + len(items))  # 正向
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    for i, related_items in sim_item.items():
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)

    return sim_item_corr


def recall_itemcf(sim_item_corr, user_items_dict, user_id, top_k, item_num):
    rank = {}
    if user_id not in user_items_dict:  # 如果不存在点击历史，冷启动, 是否存在用户画像，后者根据点击时间使用更短时间段范围的topk填充
        return []
    interacted_items = user_items_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(),key=lambda d: d[1], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.75 ** loc)

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]


def usercf(df_):
    df = df_.copy()
    # item_users=df.groupby(['item_id'])['user_id'].agg({'user_id': lambda x: list(x)}).reset_index
    item_users = df.groupby(['item_id'])['user_id'].agg(list).reset_index()
    item_users_dict = dict(zip(item_users['item_id'], item_users['user_id']))
    C = dict()
    N = defaultdict(lambda: 0)
    for i, users in item_users_dict.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if u not in C:
                    C[u] = {}
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
    interacted_items = user_items_dict[user_id]
    if sim_user_corr[user_id] == {}:
        return []
    users_sim = sim_user_corr[user_id]
    for related_user, sim in sorted(users_sim.items(), key=lambda x: x[1], reverse=True)[
                             :top_k]:  # related_user相似用户, sim相似度
        for item in user_items_dict[related_user]:
            if item in interacted_items:
                continue
            item_score_dict[item] += sim
    return sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)[:item_num]


now_phase = 9
user_path = os.path.expanduser('~')
train_path = os.path.join(user_path, r'kdd\data\underexpose_train')
test_path = os.path.join(user_path, r'kdd\data\underexpose_test')
recom_item = []
whole_click = pd.DataFrame()
click = pd.DataFrame()
times = []
item_deg = defaultdict(lambda: 0)
interacted_user_items_dict = dict()
recall_num = 500

num_user_test = 0
num_user_val=0
item_heat_lists=[]

for c in range(7, now_phase + 1):

    user_lists = []
    item_lists = []
    item_sim_lists = []
    label_lists = []
    item_rank_lists = []

    print('phase:', c)
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
    click['pred'] = click['user_id'].map(lambda x: 'test' if x in user_test else 'train')

    underline_test = click[click['pred'] == 'test'].drop_duplicates(['user_id'], keep='last')  # 当前阶段线下测试集click数据
    underline_train_val = click.append(underline_test).drop_duplicates(keep=False)  # 当前阶段以及之前阶段训练集click数据
    print('underline_test.shape:', underline_test.shape)
    print('underline_train_val.shape:', underline_train_val.shape)
    assert underline_train_val.shape[0] + underline_test.shape[0] == click.shape[0]

    underline_train_val = underline_train_val.sort_values('time')
    underline_val = underline_train_val.drop_duplicates(subset='user_id', keep='last')
    underline_train = underline_train_val.append(underline_val).drop_duplicates(keep=False)
    assert underline_train_val.shape[0] == underline_val.shape[0] + underline_train.shape[0]
    num_user_val += underline_val.shape[0]

    print('underline_val.shape:', underline_val.shape)
    print('underline_train.shape:', underline_train.shape)

    user_itemlist = underline_train_val.groupby('user_id')['item_id'].agg(list).reset_index()
    user_items_dict = dict(zip(user_itemlist['user_id'], user_itemlist['item_id']))
    with open('..\\test_data\\dict_test_user_items_phase_%d.txt' % c,'w') as fin:
        temp_str=str(user_items_dict)
        fin.write(temp_str)
    #
    item_userlist=underline_train_val.groupby('item_id')[['user_id']].agg(list).reset_index()
    dict_item_userlist=dict(zip(item_userlist['item_id'], item_userlist['user_id']))
    with open('..\\test_data\\dict_test_item_users_phase-%d.txt' % c,'w') as fin:
        temp_str=str(dict_item_userlist)
        fin.write(temp_str)
    #
    top500_click = list(underline_train_val['item_id'].value_counts().index[:500].values)  # 最热商品
    with open('..\\test_data\\item_topk500_phase-%d' % c, 'w') as f:
        temp_str=str(top500_click)
        f.write(temp_str)
    #
    item_sim_corr = itemcf(underline_train_val, user_items_dict, use_iif=False)

    # user_sim_corr = usercf(underline_train_val)
    predictions=dict()

    for index, row in tqdm(underline_test.iterrows()):

        user_list = []
        item_list = []
        item_sim_list = []
        label_list = []
        item_rank_list=[]
        # item_heat_list=[]
        # user_heat_list=[]

        user = row['user_id']
        target_item = row['item_id']


        rank_item_0 = recall_itemcf(item_sim_corr, user_items_dict, user, 500, recall_num)
        # rank_item_1 = recall_usercf(user_sim_corr, user_items_dict, user, 500, 500)
        rank=1
        if rank_item_0:
            for item in rank_item_0:
                if item[0] not in item_list:
                    user_list.append(user)
                    item_list.append(item[0])
                    item_sim_list.append(item[1])
                    item_rank_list.append(rank)
                    # item_heat_list.append(len(dict_item_userlist[item[0]]))
                    label_list.append(int(item[0] == target_item))
                    rank+=1


        # for item in rank_item_1:
        #     if item[0] not in item_list:
        #         item_list.append(item[0])

        # if len(item_list) < recall_num:
        #     for i, item in enumerate(top500_click):
        #         if item not in item_list:
        #             user_list.append(user)
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
        # item_heat_lists+=item_heat_list

    temp_dict = dict()
    temp_dict['user_id'] = user_lists
    temp_dict['item_id'] = item_lists
    temp_dict['itemcf_rank']=item_rank_lists
    # temp_dict['item_heat']=item_heat_lists
    temp_dict['itemcf_score']=item_sim_lists
    temp_dict['label']=label_lists

    df = pd.DataFrame(data=temp_dict)
    print(df.shape)
    assert df.shape[1] == 5
    print(df.head())
    df.to_csv('..\\test_data\\test_itemcf_phase-%d.csv' % c, index=False, header=False)


