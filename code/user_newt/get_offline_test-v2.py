'''
构建线下测试集underline_test
选择每阶段underexpose_test_click.csv中user_id在当前阶段以及之前阶段所有click数据中的最后一次点击作为测试集
'''
from collections import defaultdict
import pandas as pd
import os
import numpy as np
import csv
now_phase = 6
train_path = os.path.join('../../data/underexpose_train')
test_path = os.path.join('../../data/underexpose_test')
click=pd.DataFrame()
item_deg = defaultdict(lambda: 0) #item_deg是一个字典key=item_id, value=item_id的数量
# answer_fname文件存储每个阶段线下测试集数据，格式phase_id, user_id, item_id, item_deg[item_id]
answer_fname=r'../../user_data/debias_track_answer_%d.csv'




def _create_answer_file_for_evaluation(phase_id,answer_fname):
    train_test = user_path+'offline_train-%d.csv'

    # underexpose_test_qtime-T.csv contains only <user_id, time>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    # answer = 'underexpose_test_qtime_with_answer-%d.csv'  # not released
    answer=user_path+'offline_test-%d.csv'
    item_deg = defaultdict(lambda: 0)
    with open(answer_fname % phase_id, 'w') as fout:
        with open(train_test % phase_id) as fin:
            for line in fin:
                user_id, item_id, timestamp = line.split(',')
                user_id, item_id, timestamp = (
                    int(user_id), int(item_id), float(timestamp))
                item_deg[item_id] += 1
        with open(answer % phase_id) as fin:
            for line in fin:
                user_id, item_id, timestamp = line.split(',')
                user_id, item_id, timestamp = (
                    int(user_id), int(item_id), float(timestamp))
                assert user_id % 11 == phase_id
                print(phase_id, user_id, item_id, item_deg[item_id],
                      sep=',', file=fout)


def evaluate_each_phase(predictions, answers):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        if user_id in predictions.keys():
            while rank < 50 and predictions[user_id][rank] != item_id:
                rank += 1
            num_cases_full += 1.0
            if rank < 50:
                ndcg_50_full += 1.0 / np.log2(rank + 2.0)
                hitrate_50_full += 1.0
            if item_degree <= median_item_degree:
                num_cases_half += 1.0
                if rank < 50:
                    ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                    hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half

    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)

def get_rec():
    pass
answers = [{} for _ in range(now_phase+1)]
for p in range(0, now_phase + 1):
    print('phase:', p)

    click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(p), header=None, nrows=None,
                              names=['user_id', 'item_id', 'time'],
                              dtype={'user_id':np.int, 'item_id':np.int, 'time':np.float32})
    click_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(p, p), header=None, nrows=None,
                             names=['user_id', 'item_id', 'time'],
                             dtype={'user_id': np.int, 'item_id': np.int, 'time': np.float32})
    user_test=set(click_test['user_id'])  # 每阶段线下测试集用户集合
    print('len(user_val):', len(user_test))



    click_train_test = click_train.append(click_test)
    click=click.append(click_train_test,sort=True)  # 当前阶段以及之前阶段click数据
    click = click.sort_values('time')       # 时间排序
    click= click.drop_duplicates(['user_id', 'item_id', 'time'], keep='last') # 其他phase数据加入去重
    click['pred'] = click['user_id'].map(lambda x: 'test' if x in user_test else 'train') #这样的话相当于只将当前阶段的最后一次设置为test，其他都是train的数据，即这样子test中只有最后一个phase最后依次点击的数据

    offline_test = click[click['pred'] == 'test'].drop_duplicates(['user_id'], keep='last')  # 只保留最后一次点击
    offline_train = click.append(offline_test).drop_duplicates(keep=False)  # 当前阶段以及之前阶段训练集click数据
    assert (offline_test['user_id'].count() + offline_train['user_id'].count() == click['user_id'].count())
    user_path='../../user_data/offline/'
    offline_train=offline_train.reindex(['user_id', 'item_id', 'time', 'pred'], axis=1)
    offline_test=offline_test.reindex(['user_id', 'item_id', 'time', 'pred'], axis=1)
    offline_train.drop(['pred'],axis=1).to_csv(user_path +'offline_train-{}.csv'.format(p), index=False, header=False)
    offline_test.drop(['pred'],axis=1).to_csv(user_path +'offline_test-{}.csv'.format(p), index=False, header=False)

    #gen fake answer debias track answer file
    answer_fname='../../prediction_result/debias_track_answer-%d.csv'
    _create_answer_file_for_evaluation(p, answer_fname)

    #gen fake answer dict for now_p

    with open(answer_fname % p, 'r') as fin:
        for line in fin:
            line = [int(x) for x in line.split(',')]
            phase_id, user_id, item_id, item_degree = line
            assert user_id % 11 == phase_id
            # exactly one test case for each user_id
            answers[phase_id][user_id] = (item_id, item_degree)

    # pred_df= get_rec(offline_train,offline_test.user_id.unique())
pred_path='../../prediction_result/'
pred_df=pd.read_csv(pred_path+'recall_allvec_df.csv',low_memory=False, quoting=csv.QUOTE_NONE, error_bad_lines=False)
pred_df=pred_df.loc[pred_df['rank'] <=50 ]
preds={}
for user in pred_df.user_id.unique():
    assert len(pred_df.loc[pred_df.user_id == user ]) == 50
    # try:
    #     index_list=pred_df.loc[pred_df.user_id == user ].index[:50]
    #     preds[int(user)] = pred_df.iloc[index_list].item_id_pred.astype(int).to_list()
    # except IOError:
    #     print( len(pred_df.loc[pred_df.user_id == user ]) == 50, 'user id',user)
    index_list = pred_df.loc[pred_df.user_id == user].index[:50] # it returns row labels
    preds[int(user)] = pred_df.loc[index_list].item_id_pred.astype(int).to_list()

user_to_be_filled= set(preds.keys())-set(answers.keys())


evaluate_score = evaluate_each_phase(predictions=preds,answers=answers[p])
print("------------- eval result -------------")
print("hitrate_50_full : ", evaluate_score[2],'\n','  ndcg_50_full : ', evaluate_score[0], '\n')
print("hitrate_50_half : ", evaluate_score[3],'\n','  ndcg_50_half : ', evaluate_score[1], '\n')
print("score:",evaluate_score[0],'\n')



'''
下面这段不行，item_id的个数会被重复计算，因为此时temp_是累积phase的数据
比如 对于phase=1， item_deg[1]=4是phase0的结果
但是此时click train test数据集中并没有itemid==1的数据
但是agg出的数据依然是4，那么item_deg[1]就被累积更新为8
这是错误的
'''

# # 获取当前阶段以及之前阶段item_deg字典，item_id对应的点击次数，用作ndcg_topk_half和hitrate_topk_half统计
    # temp_=click
    # temp_=temp_.groupby(['item_id'], as_index=False).agg({'user_id':'count'})
    # temp_.columns=['item_id','count']
    # for item_id,item_count in zip(list(temp_['item_id'].values), list(temp_['count'].values)):
    #     item_deg[item_id]+=item_count


'''
同理下面这段也是错误的
'''
    # with open(answer_fname % c, 'w') as fout:
    #     for i, row in underline_test.iterrows():
    #         user_id, item_id, timestamp = (int(row[0]), int(row[1]), row[2])
    #         assert user_id % 11 == c
    #         print(c, user_id, item_id, item_deg[item_id], sep=',', file=fout)
    #






