import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

def topk_recall_association_rules_open_source(click_all, dict_label, k=100):
    """
        author: 青禹小生 鱼遇雨欲语与余
        修改：Cookly
        关联矩阵：
    """
    from collections import Counter

    group_by_col, agg_col = 'user_id', 'item_id'

    # data_ = click_all.groupby(['user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()

    # 分组合并同一个user_id的item_id和time项
    data_ = click_all.groupby(['user_id'])[['item_id', 'time']].agg(
        {'item_id': lambda x: ','.join(list(x)), 'time': lambda x: ','.join(list(x))}).reset_index()

    hot_list = list(click_all['item_id'].value_counts().index[:].values)
    stat_cnt = Counter(list(click_all['item_id']))
    stat_length = np.mean([len(item_txt.split(',')) for item_txt in data_['item_id']])

    matrix_association_rules = {}
    print('------- association rules matrix 生成 ---------')
    for i, row in tqdm(data_.iterrows()):

        list_item_id = row['item_id'].split(',')
        list_time = row['time'].split(',')
        len_list_item = len(list_item_id)

        for i, (item_i, time_i) in enumerate(zip(list_item_id, list_time)):
            for j, (item_j, time_j) in enumerate(zip(list_item_id, list_time)):

                t = np.abs(float(time_i) - float(time_j))
                d = np.abs(i - j)

                if i < j:
                    if item_i not in matrix_association_rules:
                        matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                        matrix_association_rules[item_i][item_j] = 0

                    matrix_association_rules[item_i][item_j] += 1 * 0.7 * (0.8 ** (d - 1)) * (1 - t * 10000) / np.log(
                        1 + len_list_item)

                if i > j:
                    if item_i not in matrix_association_rules:
                        matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                        matrix_association_rules[item_i][item_j] = 0

                    matrix_association_rules[item_i][item_j] += 1 * 1.0 * (0.8 ** (d - 1)) * (1 - t * 10000) / np.log(
                        1 + len_list_item)

    # print(len(matrix_association_rules.keys()))
    # print(len(set(click_all['item_id'])))
    # print('data - matrix: ')
    # print( set(click_all['item_id']) - set(matrix_association_rules.keys()) )
    # print('matrix - data: ')
    # print( set(matrix_association_rules.keys()) - set(click_all['item_id']))
    assert len(matrix_association_rules.keys()) == len(set(click_all['item_id']))

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- association rules 召回 ---------')
    for i, row in tqdm(data_.iterrows()):

        list_item_id = row['item_id'].split(',')

        dict_item_id_score = {}
        for i, item_i in enumerate(list_item_id[::-1]): #list_item_id[::-1]为什么要倒序
            sorted_items=sorted(matrix_association_rules[item_i].items(), reverse=True)
            for item_j, score_similar in sorted_items[0:k]:
                if item_j not in list_item_id: # 物品不在用户购买历史中
                    if item_j not in dict_item_id_score:
                        dict_item_id_score[item_j] = 0

                    dict_item_id_score[item_j] += score_similar * (0.7 ** i)

        # 对dict_item_id_score 按相似度分数得分排序
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])

        # 不足的热度补全
        if len(dict_item_id_score_topk) < k:
            for i, item in enumerate(hot_list):
                if (item not in list_item_id) and (item not in dict_item_id_set):
                    item_similar = item
                    score_similar = - i - 100
                    dict_item_id_score_topk.append((item_similar, score_similar))
                if len(dict_item_id_score_topk) == k:
                    break

        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])

    topk_recall = pd.DataFrame(
        {'user_id': list_user_id, 'item_similar': list_item_similar, 'score_similar': list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall


def metrics_recall(topk_recall, phase, k, sep=10):
    data_ = topk_recall[topk_recall['pred'] == 'train'].sort_values(['user_id', 'score_similar'], ascending=False)
    data_ = data_.groupby(['user_id']).agg(
        {'item_similar': lambda x: list(x), 'next_item_id': lambda x: ''.join(set(x))})

    data_['index'] = [recall_.index(label_) if label_ in recall_ else -1 for (label_, recall_) in
                      zip(data_['next_item_id'], data_['item_similar'])]

    print('-------- 召回效果 -------------')
    print('--------:phase: ', phase, ' -------------')
    data_num = len(data_)
    for topk in range(0, k + 1, sep):
        hit_num = len(data_[(data_['index'] != -1) & (data_['index'] <= topk)])
        hit_rate = hit_num * 1.0 / data_num
        print('phase: ', phase, ' top_', topk, ' : ', 'hit_num : ', hit_num, 'hit_rate : ', hit_rate, ' data_num : ',
              data_num)
        print()

    hit_rate = len(data_[data_['index'] != -1]) * 1.0 / data_num
    return hit_rate




now_phase = 6
train_path = os.path.join(os.path.expanduser('~'), 'kdd\\data\\underexpose_train')
test_path = os.path.join(os.path.expanduser('~'), 'kdd\\data\\underexpose_test')

# train
flag_append = True
flag_test = False
recall_num = 500
topk = 50
nrows = None

# test
# flag_append = False
# flag_test = True
# recall_num = 50
# topk = 50
# nrows = 1000


submit_all = pd.DataFrame()
click_all = pd.DataFrame()
for phase in range(0, now_phase + 1):
    print('phase:', phase)
    click_train = pd.read_csv(
        os.path.join(train_path, 'underexpose_train_click-{phase}.csv'.format(phase=phase))
        , header=None
        , nrows=nrows
        , names=['user_id', 'item_id', 'time']
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
    )
    click_test = pd.read_csv(
        os.path.join(test_path,'underexpose_test_click-{phase}.csv'.format(phase=phase))
        , header=None
        , nrows=nrows
        , names=['user_id', 'item_id', 'time']
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
    )

    click = click_train.append(click_test)

    if flag_append:
        click_all = click_all.append(click)
    else:
        click_all = click

    click_all = click_all.sort_values('time')
    click_all = click_all.drop_duplicates(['user_id', 'item_id', 'time'], keep='last')

    set_pred = set(click_test['user_id'])
    set_train = set(click_all['user_id']) - set_pred


    temp_ = click_all
    temp_['pred'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')
    temp_ = temp_[temp_['pred'] == 'train'].drop_duplicates(['user_id'], keep='last')
    temp_['remove'] = 'remove'
    # print('temp_: ',temp_['user_id'].count())
    # print(temp_.head())


    train_test = click_all
    # print(train_test.head())
    train_test = train_test.merge(temp_, on=['user_id', 'item_id', 'time', 'pred'], how='left')
    train_test = train_test[train_test['remove'] != 'remove']


    dict_label_user_item = dict(zip(temp_['user_id'], temp_['item_id']))  # 线下验证集

    temp_ = train_test.groupby(['item_id'])['user_id'].count().reset_index()
    temp_ = temp_.sort_values(['item_id'])
    hot_list = list(temp_['item_id'])[::-1]

    print('-------- 召回 -------------')
    topk_recall = topk_recall_association_rules_open_source(
        click_all=train_test
        , dict_label=dict_label_user_item
        , k=recall_num
    )

    print('-------- 评测召回效果 -------------')
    hit_rate = metrics_recall(topk_recall=topk_recall, phase=phase, k=recall_num, sep=int(recall_num / 10))
    print('-------- 召回TOP:{k}时, 命中百分比:{hit_rate} -------------'.format(k=recall_num, hit_rate=hit_rate))

