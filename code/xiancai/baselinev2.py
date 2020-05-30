import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math
import numpy as np
import time


def get_sim_item(all_click, user_col, item_col, time_col, use_iuf=False, use_time=False):
    # 生成用户的点击商品序列，一行代表一个用户的点击序列
    user_item = all_click.groupby(user_col)[item_col].agg(list).reset_index()
    # 生成字典
    user_item_dict = dict(zip(user_item[user_col], user_item[item_col]))
    # 生成用户的点击时间序列
    user_time = all_click.groupby(user_col)[time_col].agg(
        list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time[user_col], user_time[time_col]))

    sim_item = {}  # 保存商品相似度
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):
        # 被同一个用户点击的商品
        for ixi, i in enumerate(items):  # ixi是loc1，i是item
            item_cnt[i] += 1
            sim_item.setdefault(i, {})

            for ixj, j in enumerate(items):  # ixj是loc2，j是relate_item
                if i == j:
                    continue
                sim_item[i].setdefault(j, 0)

                t1 = user_time_dict[user][ixi]  # 点击时间提取
                t2 = user_time_dict[user][ixj]

                if not use_iuf:
                    sim_item[i][j] += 1
                else:
                    if not use_time:
                        sim_item[i][j] += 1 / math.log(1 + len(items))
                    else:
                        flag = True if ixi > ixj else False
                        num = max(ixi - ixj, ixj - ixi) - 1
                        t = max(t1 - t2, t2 - t1) * 10000
                        indicator = 1.0 if flag else 1.5
                        sim_item[i][j] += 1.0 * indicator * \
                            (0.8 ** num) * (1 - t) / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    for i, related_items in tqdm(sim_item.items()):
        for j, sim in related_items.items():
            #sim_item_corr[i][j] = sim / math.sqrt(item_cnt[i] * item_cnt[j])
            sim_item_corr[i][j] = sim / (item_cnt[i] * item_cnt[j]) ** 0.2

    return sim_item_corr, user_item_dict

# 交互行为打分，根据位置远近添加权重


def recommend(item_sim_list, user_item_dict, user_id, topk, item_num):

    user_items = user_item_dict[user_id]
    user_items = user_items[::-1]
    rank = {}
    for ixi, i in enumerate(user_items):
        a=sorted(item_sim_list[i].items(), key=lambda x: x[1], reverse=True)
        for j, sim in sorted(item_sim_list[i].items(), key=lambda x: x[1], reverse=True)[: topk]:
            if j not in user_items:
                rank.setdefault(j, 0)
                rank[j] += sim * (0.75 ** ixi)

    return sorted(rank.items(), key=lambda x: x[1], reverse=True)[: item_num]

# fill user to 50 items，召回的50个商品中，是最后一次点击的商品label是1，反之为0.


def get_predict(rec_df, pred_col, top_50_clicks):
    top_50_clicks = [int(t) for t in top_50_clicks.split(',')]
    scores = [-1 * (i + 1) for i in range(0, len(top_50_clicks))]
    ids = list(rec_df['user_id'].unique())

    fill_df = pd.DataFrame(ids * len(top_50_clicks), columns=['user_id'])
    fill_df.sort_values('user_id', inplace=True)
    fill_df['item_id'] = top_50_clicks * len(ids)
    fill_df[pred_col] = scores * len(ids)
    rec_df = rec_df.append(fill_df)
    rec_df.sort_values(pred_col, ascending=False, inplace=True)
    rec_df = rec_df.drop_duplicates(
        subset=['user_id', 'item_id'], keep='first')
    rec_df['rank'] = rec_df.groupby('user_id')[pred_col].rank(
        method='first', ascending=False)
    rec_df = rec_df[rec_df['rank'] <= 50]
    rec_df = rec_df.groupby('user_id')['item_id'].apply(lambda x: ','.join(
        [str(i) for i in x])).str.split(',', expand=True).reset_index()

    return rec_df


if __name__ == '__main__':
    current_phase = 6
    train_path = '../../data/underexpose_train/'
    test_path = '../../data/underexpose_test/'
    rec_items = []

    whole_click = pd.DataFrame()
    for phase in range(current_phase + 1):
        print("phase: ", phase)
        train_click = pd.read_csv(train_path + 'underexpose_train_click-{}.csv'.format(
            phase), header=None, names=['user_id', 'item_id', 'time'])
        test_click = pd.read_csv(test_path + 'underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(
            phase,phase), header=None, names=['user_id', 'item_id', 'time'])
        test_users = pd.read_csv(test_path + 'underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(
            phase,phase), header=None, names=['user_id', 'time'])

        all_click = train_click.append(test_click)
        whole_click = whole_click.append(all_click)
        whole_click = whole_click.drop_duplicates(
            subset=['user_id', 'item_id', 'time'], keep='last')
        whole_click = whole_click.sort_values('time')  # get_sim_click的输入

        item_sim_list, user_item = get_sim_item(
            whole_click, 'user_id', 'item_id', 'time', use_iuf=True, use_time=True)

        # qtime，unique()返回参数数组中所有不同的数并从小到大排序
        for i in tqdm(test_users['user_id'].unique()):
            rank_items = recommend(
                item_sim_list, user_item, i, 500, 50)  # 推荐50
            for j in rank_items:
                rec_items.append([i, j[0], j[1]])

    '''
    test_users = pd.read_csv(test_path + 'underexpose_test_qtime-{}.csv'.format(current_phase), header = None, names = ['user_id','time'])

    for i in tqdm(test_users['user_id'].unique()):
            rank_items = recommend(item_sim_list, user_item, i, 500, 50)
            for j in rank_items:
                rec_items.append([i, j[0], j[1]])
    '''
    # find most popular items for cold-start users
    top_50_clicks = whole_click['item_id'].value_counts().index[:50].values
    top_50_clicks = ','.join([str(i) for i in top_50_clicks])

    rec_df = pd.DataFrame(rec_items, columns=['user_id', 'item_id', 'sim'])
    result = get_predict(rec_df, 'sim', top_50_clicks)

    # prediction_dict = {}
    # for i, row in result.iterrows():
    #     user_id = row['user_id']
    #     item_list = [row[num] for num in range(50)]
    #     prediction_dict[user_id] = item_list
    #
    #
    # def evaluate_each_phase(predictions, answer):
    #     list_item_degress = []
    #     for i, row in answer.iterrows():
    #         item_degree = row['item_deg']
    #         list_item_degress.append(item_degree)
    #     list_item_degress.sort()
    #     median_item_degree = list_item_degress[len(list_item_degress) // 2]
    #
    #     num_cases_full = 0.0
    #     ndcg_50_full = 0.0
    #     ndcg_50_half = 0.0
    #     num_cases_half = 0.0
    #     hitrate_50_full = 0.0
    #     hitrate_50_half = 0.0
    #     for i, row in tqdm(answer.iterrows()):
    #         user_id = row['user_id']
    #         item_id = row['item_id']
    #         item_degree = row['item_deg']
    #         rank = 0
    #
    #         while (rank < 50) and (predictions[int(user_id)][rank] != item_id):
    #             rank += 1
    #         num_cases_full += 1.0
    #         if rank < 50:
    #             ndcg_50_full += 1.0 / np.log2(rank + 2.0)
    #             hitrate_50_full += 1.0
    #         if item_degree <= median_item_degree:
    #             num_cases_half += 1.0
    #             if rank < 50:
    #                 ndcg_50_half += 1.0 / np.log2(rank + 2.0)
    #                 hitrate_50_half += 1.0
    #     ndcg_50_full /= num_cases_full
    #     hitrate_50_full /= num_cases_full
    #     ndcg_50_half /= num_cases_half
    #     hitrate_50_half /= num_cases_half
    #     return np.array([ndcg_50_full, ndcg_50_half,
    #                      hitrate_50_full, hitrate_50_half], dtype=np.float32)
    #
    #
    # evaluate_score = np.zeros(4, dtype=np.float32)
    # for phase in range(now_phase + 1):
    #     answer = all_click_test[all_click_test['phase'] == phase]
    #     evaluate_score += evaluate_each_phase(prediction_dict, answer)
    # print("------------- eval result -------------")
    # print("hitrate_50_full : ", evaluate_score[2], '\n', '  ndcg_50_full : ', evaluate_score[0], '\n')
    # print("hitrate_50_half : ", evaluate_score[3], '\n', '  ndcg_50_half : ', evaluate_score[1], '\n')
    # print("score:", evaluate_score[0], '\n')

    time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    file_name = '../../prediction_result/underexpose_submit{time_str}.csv'.format(time_str=time_str)
    result.to_csv(file_name, index=False, header=None)

