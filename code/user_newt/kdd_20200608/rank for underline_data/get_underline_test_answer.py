'''
构建线下测试集underline_test
选择每阶段underexpose_test_click.csv中user_id在当前阶段以及之前阶段所有click数据中的最后一次点击作为测试集
'''
from collections import defaultdict
import pandas as pd
import os
import numpy as np
now_phase = 9
user_path=os.path.expanduser('~')   #用户目录
train_path = os.path.join(user_path, r'kdd\data\underexpose_train')
test_path = os.path.join(user_path, r'kdd\data\underexpose_test')
click=pd.DataFrame()
item_deg = defaultdict(lambda: 0) #item_deg是一个字典key=item_id, value=item_id的数量
# answer_fname文件存储每个阶段线下测试集数据，格式phase_id, user_id, item_id, item_deg[item_id]
answer_fname='..\\data\\debias_track_answer_%d.csv'


for c in range(0, now_phase + 1):
    print('phase:', c)
    click_train = pd.read_csv(train_path + '\\underexpose_train_click-{}.csv'.format(c), header=None,nrows=None,
                              names=['user_id', 'item_id', 'time'],
                              dtype={'user_id':np.int, 'item_id':np.int, 'time':np.float32})
    click_test = pd.read_csv(test_path + '\\underexpose_test_click-{}.csv'.format(c), header=None,nrows=None,
                             names=['user_id', 'item_id', 'time'],
                             dtype={'user_id': np.int, 'item_id': np.int, 'time': np.float32})

    user_test=set(click_test['user_id'])  # 每阶段线下测试集用户集合
    print('len(user_test):', len(user_test))


    click_train_test = click_train.append(click_test)
    click=click.append(click_train_test)  # 当前阶段以及之前阶段click数据
    click = click.sort_values('time')       # 时间排序
    click= click.drop_duplicates(['user_id', 'item_id', 'time'], keep='last') # 去重

    if c<7:
        continue
    # 获取当前阶段以及之前阶段item_deg字典，item_id对应的点击次数，用作ndcg_topk_half和hitrate_topk_half统计
    temp_=click
    temp_=temp_.groupby(['item_id'], as_index=False).agg({'user_id':'count'})
    temp_.columns=['item_id','count']
    for item_id,item_count in zip(list(temp_['item_id'].values), list(temp_['count'].values)):
        item_deg[item_id]+=item_count


    click['pred'] = click['user_id'].map(lambda x: 'test' if x in user_test else 'train')
    underline_test=click[click['pred']=='test'].drop_duplicates(['user_id'], keep='last') #当前阶段线下测试集click数据
    underline_train_val = click.append(underline_test).drop_duplicates(keep=False)  #当前阶段以及之前阶段训练集click数据
    print('underline_test:', underline_test['user_id'].count())
    assert (underline_test['user_id'].count()+underline_train_val['user_id'].count()==click['user_id'].count())


    # underline_test
    with open(answer_fname % c, 'w') as fout:
        for i, row in underline_test.iterrows():
            user_id, item_id, timestamp = (int(row[0]), int(row[1]), row[2])
            assert user_id % 11 == c
            print(c, user_id, item_id, item_deg[item_id], sep=',', file=fout)




