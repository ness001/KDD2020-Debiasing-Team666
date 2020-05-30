from collections import defaultdict
import pandas as pd
import math
import os
import numpy as np

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
                        sim_item[item][relate_item] += 1 * 1.0  * (0.8**(loc1-loc2-1)) *(
                                    1 - (t1 - t2) * 10000) / math.log(1 + len(items))  # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0  * (0.8**(loc2-loc1-1)) * (
                                    1 - (t2 - t1) * 10000) / math.log(1 + len(items))  # 正向
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    for i, related_items in sim_item.items():
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)

    return sim_item_corr


def recall_itemcf(sim_item_corr, user_items_dict, user_id, top_k, item_num):
    '''
    input:item_sim_list, user_item, uid, 500, 50
    # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
    '''
    rank = {}
    interacted_items = user_items_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.75 ** loc)

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]


now_phase = 6
user_path=os.path.expanduser('~')
train_path = os.path.join(user_path, r'kdd\data\underexpose_train')
test_path = os.path.join(user_path, r'kdd\data\underexpose_test')
recom_item = []
whole_click = pd.DataFrame()
click=pd.DataFrame()
times=[]
answer_fname=r'data\\debias_track_answer_%d.csv'
item_deg = defaultdict(lambda: 0)

for c in range(0,now_phase + 1):
    print('phase:', c)
    click_train = pd.read_csv(train_path + '\\underexpose_train_click-{}.csv'.format(c), header=None,nrows=None,
                              names=['user_id', 'item_id', 'time'],
                              dtype={'user_id':np.int, 'item_id':np.int, 'time':np.float32})
    click_test = pd.read_csv(test_path + '\\underexpose_test_click-{}.csv'.format(c), header=None,nrows=None,
                             names=['user_id', 'item_id', 'time'],
                             dtype={'user_id': np.int, 'item_id': np.int, 'time': np.float32})

    user_test=set(click_test['user_id'])     # 每阶段线下测试集用户集合
    print('len(user_val):', len(user_test))
    user_train=set(click_train['user_id'])-user_test  #每阶段训练集用户集合


    click_train_test = click_train.append(click_test)
    click=click.append(click_train_test)        # 当前阶段以及之前阶段click数据
    click = click.sort_values('time')           # 时间排序
    click= click.drop_duplicates(['user_id', 'item_id', 'time'], keep='last') # 去重


    click['pred'] = click['user_id'].map(lambda x: 'test' if x in user_test else 'train')
    underline_test=click[click['pred']=='test'].drop_duplicates(['user_id'], keep='last') #当前阶段线下测试集click数据
    underline_train=click.append(underline_test).drop_duplicates(keep=False)  #当前阶段以及之前阶段训练集click数据

    user_item_ = underline_train.groupby('user_id')['item_id'].agg(list).reset_index()
    # user_items_dict是一个字典，key=user_id,value=用户点击的item列表
    user_items_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))

    top500_click = underline_train['item_id'].value_counts().index[:500].values  # 最热商品
    hot_list = list(top500_click)

    item_sim_list = itemcf(underline_train, user_items_dict, use_iif=False) #计算item的相似度矩阵
    predictions = {}

    for index, user in enumerate(user_test):
        rank_item = recall_itemcf(item_sim_list, user_items_dict, user, 500, 100) #召回100个
        item_list = []
        for item in rank_item:
            item_list.append(item[0])
        if len(item_list) < 100:
            for i, item in enumerate(hot_list):
                if item not in item_list:
                    item_list.append(item)
                if len(item_list) == 100:
                    break
        predictions[user] = item_list

    with open('data\\underexpose_submit-%d.csv' % c, 'w') as f:
        temp = str(predictions)
        f.write(temp)



