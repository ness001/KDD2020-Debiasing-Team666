def topk_recall_word2vec_embedding(click_all, dict_label, k=100, dim=64, epochs=40, learning_rate=0.5):
    import gensim

    data_ = click_all.groupby(['pred', 'user_id'])['item_id'].agg(lambda x: ','.join(list(x))).reset_index()
    list_data = list(data_['item_id'].map(lambda x: x.split(',')))

    model = gensim.models.Word2Vec(list_data,
                                   size=dim,
                                   alpha=learning_rate,
                                   window=999999,
                                   min_count=1,
                                   workers=4,
                                   compute_loss=True,
                                   iter=epochs,
                                   hs=0,
                                   sg=1,
                                   seed=42)
    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- word2vec 召回 ---------')
    for i, row in tqdm(data_.iterrows()):

        list_item_id = row['item_id'].split(',')
        dict_item_id_score = {}

        for i, item in enumerate(list_item_id[::-1]):
            most_topk = model.wv.most_similar(item, topn=k)

            for item_similar, score_similar in most_topk:
                if item_similar not in list_item_id:
                    if item_similar not in dict_item_id_score:
                        dict_item_id_score[item_similar] = 0
                    sigma = 0.8
                    dict_item_id_score[item_similar] += 1.0 / (1 + sigma * i) * score_similar

        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]

        assert len(dict_item_id_score_topk) == k

        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])

        assert len(dict_item_id_set) == k

        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])

    topk_recall = pd.DataFrame({'user_id': list_user_id,
                                'item_similar': list_item_similar,
                                'score_similar': list_score_similar})

    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)

    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall

def metrics_recall(topk_recall, phase, k, sep=10):
    data_ = topk_recall.sort_values(['user_id','score_similar'],ascending=False)
    data_ = data_.groupby(['user_id']).agg({'item_similar':lambda x:list(str(x)),'next_item_id':lambda x:''.join(set(str(x)))})

    data_['index'] = [recall_.index(label_) if label_ in recall_ else -1 for (label_, recall_) in zip(data_['next_item_id'],data_['item_similar'])]

    print('-------- 召回效果 -------------')
    print('--------:phase: ', phase,' -------------')
    data_num = len(data_)
    for topk in range(0,k+1,sep):
        hit_num = len(data_[(data_['index']!=-1) & (data_['index']<=topk)])
        hit_rate = hit_num * 1.0 / data_num
        print('phase: ', phase, ' top_', topk, ' : ', 'hit_num : ', hit_num, 'hit_rate : ', hit_rate, ' data_num : ', data_num)
        print()

    hit_rate = len(data_[data_['index']!=-1]) * 1.0 / data_num
    return hit_rate

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")
now_phase = 0
user_path = os.path.expanduser('~')
train_path = os.path.join(user_path, r'kdd\data\underexpose_train')
test_path = os.path.join(user_path, r'kdd\data\underexpose_test')
flag_append = False
flag_test = False
recall_num = 500
topk = 50
nrows = None
submit_all = pd.DataFrame()
click_all = pd.DataFrame()

for phase in range(0, now_phase + 1):
    print('phase:', phase)
    click_train = pd.read_csv(train_path + r'\underexpose_train_click-{phase}.csv'.format(phase=phase),
                              header=None, nrows=None,
                              names=['user_id', 'item_id', 'time'],
                              dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str})
    click_test = pd.read_csv(test_path + r'\underexpose_test_click-{phase}.csv'.format(phase=phase),
                             header=None, nrows=None,
                             names=['user_id', 'item_id', 'time'],
                             dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str})

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

    temp_ = temp_[temp_['pred'] == 'test'].drop_duplicates(['user_id'], keep='last') #保留test_click中user的最后一次点击
    temp_['remove'] = 'remove'

    train_test = click_all
    # print(train_test.head(20))

    train_test = train_test.merge(temp_, on=['user_id', 'item_id', 'time', 'pred'], how='left')
    print(train_test.shape)

    train = train_test[train_test['remove'] != 'remove']   #移除train_click中user的最后一次点击
    train=train.drop(labels=['pred','remove'],axis=1)
    print(train.shape)
    print(train.head())

    dict_label_user_item = dict(zip(temp_['user_id'], temp_['item_id']))    #train_click中user的最后一次点击作为label
    temp_ = train.groupby(['item_id'])['user_id'].count().reset_index()
    temp_ = temp_.sort_values(['item_id'])
    hot_list = list(temp_['item_id'])[::-1]
    print('-------- 召回 -------------')
    topk_recall = topk_recall_word2vec_embedding(click_all=train_test,
                                                 dict_label=dict_label_user_item,
                                                 k=500,
                                                 dim=128,
                                                 epochs=40,
                                                 learning_rate=0.04)
    print('-------- 评测召回效果 -------------')
    hit_rate = metrics_recall(topk_recall=topk_recall, phase=phase, k=recall_num, sep=int(recall_num / 10))
    print('-------- 召回TOP:{k}时, 命中百分比:{hit_rate} -------------'.format(k=recall_num, hit_rate=hit_rate))

