'''
评估预测结果
'''
now_phase=7
# answer_fname文件存储每个阶段线下测试集数据，格式phase_id, user_id, item_id, item_deg[item_id]
answer_fname=r'..\data\debias_track_answer_%d.csv'  # 每阶段线下测试集的实际点击
submit_answer_fname='..\\test_data\\underexpose_submit-%d.csv' #每阶段线下测试集的user_id的预测点击
#TODO
# submit_answer_fname文件保存了线下测试集的user_id的预测点击
import numpy as np
from collections import defaultdict
import pandas as pd

for topk in [50, 100, 500]: #topk=50,100,...500
    total_ndcg_topk_full = 0.0
    total_ndcg_topk_half = 0.0
    total_hitrate_topk_full = 0.0
    total_hitrate_topk_half = 0.0

    for phase in range(7, now_phase+1):

        with open('..\\test_data\\item_topk500_phase-%d' %phase, 'r') as f:
            temp_str = f.read()
            list_topk500_items = eval(temp_str)

        # test_df = pd.read_csv('..\\test_data\\test_itemcf_phase-%d.csv' % phase,
        #                       header=None,
        #                       names=['user_id', 'item_id', 'itemcf_rank', 'itemcf_score', 'label'] )
        # print('\ntest_df:')
        # print(test_df.shape)
        # print(test_df.head())

        # test_df = test_df.groupby('user_id')['item_id'].agg(list)
        # print('\ntest_df user-items: ', test_df.shape)
        # print(test_df.head())
        # predictions = dict(zip(test_df.index.to_list(), test_df.values.tolist()))

        # with open(submit_answer_fname % phase, 'r') as f:
        #     temp = f.read()
        #     predictions = eval(temp)  # prediction是一个字典key=user_id vakue=预测点击的商品列表
        test_df=pd.read_csv('..\\test_data\\test_w2v_phase-%d' % phase,
                              header=None,
                              names=['user_id', 'item_id', 'itemcf_score'] )
        test_df = test_df.groupby('user_id')['item_id'].agg(list)
        predictions = dict(zip(test_df.index.to_list(), test_df.values.tolist()))
        answers={}
        with open(answer_fname % phase, 'r') as fin:
            for line in fin:
                line = [int(x) for x in line.split(',')]
                phase_id, user_id, item_id, item_degree = line
                assert user_id % 11 == phase_id
                answers[user_id] = (item_id, item_degree)

        list_item_degress = []
        for user_id in answers:
            item_id, item_degree = answers[user_id]
            list_item_degress.append(item_degree)
        list_item_degress.sort()
        median_item_degree = list_item_degress[len(list_item_degress) // 2] #商品统计次数中值

        num_cases_full = 0.0
        ndcg_topk_full = 0.0
        ndcg_topk_half = 0.0
        num_cases_half = 0.0
        hitrate_topk_full = 0.0
        hitrate_topk_half = 0.0
        
        for user in answers:
            item_id, item_degree = answers[user]
            item_list = predictions[user]
            # if len(item_list)!=50:
            #     print(len(item_list))
            # if user in predictions:
            #     item_list=predictions[user]
            # else:
            #     item_list=list_topk500_items
            rank = 0
            while rank < min(len(item_list),topk) and item_list[rank] != item_id:
                rank += 1
            num_cases_full += 1.0
            if rank < min(topk,len(item_list)):
                ndcg_topk_full += 1.0 / np.log2(rank + 2.0)
                hitrate_topk_full += 1.0
            if item_degree <= median_item_degree:
                num_cases_half += 1.0
                if rank <min(topk,len(item_list)):
                    ndcg_topk_half += 1.0 / np.log2(rank + 2.0)
                    hitrate_topk_half += 1.0

        ndcg_topk_full /= num_cases_full
        ndcg_topk_half /= num_cases_half
        hitrate_topk_full /= num_cases_full
        hitrate_topk_half /= num_cases_half

        # 每阶段指标相加
        total_ndcg_topk_full += ndcg_topk_full
        total_ndcg_topk_half += ndcg_topk_half
        total_hitrate_topk_full += hitrate_topk_full
        total_hitrate_topk_half += hitrate_topk_half

        print('topk={}\npahse={}\nndcg_topk_full={}\nndcg_topk_half={}\nhitrate_topk_full={}\nhitrate_topk_half={}\n'.format(
            topk, phase, ndcg_topk_full, ndcg_topk_half, hitrate_topk_full, hitrate_topk_half))
        with open('..\\data\\score_detail.csv', 'a+') as f:
            print('topk={}\npahse={}\nndcg_topk_full={}\nndcg_topk_half={}\nhitrate_topk_full={}\nhitrate_topk_half={}\n'
                  .format(topk, phase, ndcg_topk_full, ndcg_topk_half, hitrate_topk_full, hitrate_topk_half),file=f)

    # print('topk={}\n, ndcg_topk_full={}\n, ndcg_topk_half={}\n, hitrate_topk_full={}\n, hitrate_topk_half={}\n'.format(
    #       topk, total_ndcg_topk_full, total_ndcg_topk_half, total_hitrate_topk_full, total_hitrate_topk_half))

    # with open('..\\data\\score', 'a+') as f:
    #     print(topk, total_ndcg_topk_full, total_ndcg_topk_half, total_hitrate_topk_full, total_hitrate_topk_half,
    #           sep=',', file=f)




