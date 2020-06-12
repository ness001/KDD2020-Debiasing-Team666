import numpy as np
import pandas as pd

#TODO
start=7
now_phase=9
answer_fname='..\\data\\debias_track_answer_%d.csv'  # 每阶段线下测试集的实际点击
submit_answer_fname='..\\test_data\\test_result_2_phase-%d.csv'
topk_500_file='..\\test_data\\item_topk500_phase-%d'
output_file='..\\test_data\\score_detail.csv'

for topk in [50]: #topk=50,100,...500
    # print('topk={}\n'.format(topk))
    total_ndcg_topk_full = 0.0
    total_ndcg_topk_half = 0.0
    total_hitrate_topk_full = 0.0
    total_hitrate_topk_half = 0.0
    for phase in range(start, now_phase+1):

        test_df = pd.read_csv(submit_answer_fname % phase,
                              header=None,
                              names=['user_id', 'item_id', 'txt_sim', 'img_sim', 'itemcf_rank', 'itemcf_score',
                                     'label', 'usercf_rank', 'usercf_score', 'w2v_rank', 'w2v_score','item_heat', 'user_heat', 'score'])
        # print('\ntest_df:', test_df.shape)
        # print(test_df.head())
        test_df = test_df.groupby('user_id')[['item_id', 'score']].agg(list)

        # print('\ntest_df user-items-scores: ', test_df.shape)
        # print(test_df.head())
        prediction_user_set = set(test_df.index.to_list())

        with open(topk_500_file % phase, 'r') as f:
            temp_str = f.read()
            list_topk500_items = eval(temp_str)

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

        # with open('data\\temp.csv_%d' %phase, 'r') as fin:
        #     temp_str=fin.read()
        #     predictions=eval(temp_str)

        num_cases_full = 0.0
        ndcg_topk_full = 0.0
        ndcg_topk_half = 0.0
        num_cases_half = 0.0
        hitrate_topk_full = 0.0
        hitrate_topk_half = 0.0

        for user in answers:
            item_id, item_degree = answers[user]
            if user in prediction_user_set:
                item_list=np.array(test_df.loc[user]['item_id'])
                score_array=np.array(test_df.loc[user]['score'])
                index=np.argsort(score_array)[::-1]
                item_list=list(item_list[index])
                score_list=list(score_array[index])
                if len(item_list)<500:
                    item_list+=list_topk500_items
                item_list=item_list[:500]
            else:
                item_list=list_topk500_items
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

        with open(output_file, 'a+') as f:
            print('topk={}\npahse={}\nndcg_topk_full={}\nndcg_topk_half={}\nhitrate_topk_full={}\nhitrate_topk_half={}\n'
                .format(topk, phase, ndcg_topk_full, ndcg_topk_half, hitrate_topk_full, hitrate_topk_half), file=f)

# print('topk={}\n, ndcg_topk_full={}\n, ndcg_topk_half={}\n, hitrate_topk_full={}\n, hitrate_topk_half={}\n'.format(
#       topk, total_ndcg_topk_full, total_ndcg_topk_half, total_hitrate_topk_full, total_hitrate_topk_half))
# with open('..\\data\\score', 'a+') as f:
#     print(topk, total_ndcg_topk_full, total_ndcg_topk_half, total_hitrate_topk_full, total_hitrate_topk_half,
#           sep=',', file=f)




