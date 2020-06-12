def topk_recall_word2vec_embedding(df_for_train,u2i_for_test, phase, k=500, dim=64, epochs=40, learning_rate=0.5):
    import gensim
    print(df_for_train.head())
    df_user_items = df_for_train.groupby('user_id')['item_id'].agg(list)
    print(df_user_items.head())
    users_for_train=set(df_user_items.index.values)
    list_items = list(df_user_items.values)

    model = gensim.models.Word2Vec(list_items,
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
    list_w2v_rank=[]
    list_label=[]
    print('------- word2vec 召回 ---------')
    for user in tqdm(u2i_for_test):
        if user not in users_for_train:
            continue
        list_item_id = df_user_items.loc[user]
        dict_item_id_score = {}
        target_item=u2i_for_test[user]

        for i, item in enumerate(list_item_id[::-1]):
            most_topk = model.wv.most_similar(item, topn=k)

            for item_similar, score_similar in most_topk:
                if item_similar not in list_item_id: #相似item不在用户点击历史中
                    if item_similar not in dict_item_id_score:
                        dict_item_id_score[item_similar] = 0
                    sigma = 0.8
                    dict_item_id_score[item_similar] += 1.0 / (1 + sigma * i) * score_similar

        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]

        assert len(dict_item_id_score_topk) == k

        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])

        assert len(dict_item_id_set) == k
        rank=1
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(user)
            list_w2v_rank.append(rank)
            list_label.append(int(item_similar==target_item))
            rank+=1

    topk_recall = pd.DataFrame({'user_id': list_user_id,
                                'item_id': list_item_similar,
                                'w2v_rank': list_w2v_rank,
                                'w2v_score': list_score_similar,
                                'label': list_label})
    print(topk_recall.shape)
    topk_recall.to_csv(output_file %phase, header=False,index=False)



import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os

start = 7
end=10
user_path = os.path.expanduser('~')
train_path = os.path.join(user_path, r'kdd\data\underexpose_train')
test_path = os.path.join(user_path, r'kdd\data\underexpose_test')
output_file='..\\data\\train_w2v_last2_phase-%d.csv'
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

for c in range(start, end):

    user_lists = []
    item_lists = []
    item_sim_lists = []
    label_lists = []
    item_rank_lists = []

    print('phase:', c)
    click_train = pd.read_csv(train_path + '\\underexpose_train_click-{}.csv'.format(c), header=None, nrows=None,
                              names=['user_id', 'item_id', 'time'],
                              dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float32})
    click_test = pd.read_csv(test_path + '\\underexpose_test_click-{}.csv'.format(c), header=None, nrows=None,
                             names=['user_id', 'item_id', 'time'],
                             dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float32})

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

    underline_val_2 = underline_train.drop_duplicates(subset='user_id', keep='last')
    underline_train_2 = underline_train.append(underline_val_2).drop_duplicates(keep=False)
    assert underline_train.shape[0] == underline_val_2.shape[0] + underline_train_2.shape[0]

    underline_val = underline_val.append(underline_val_2)
    underline_train = underline_train_2

    dict_label_user2item=dict(zip(underline_val['user_id'],underline_val['item_id']))
    num_user_val += underline_val.shape[0]

    print('underline_val.shape:', underline_val.shape)
    print('underline_train.shape:', underline_train.shape)

    topk_recall_word2vec_embedding(underline_train,dict_label_user2item,c,k=recall_num,dim=128,epochs=40,learning_rate=0.04)



