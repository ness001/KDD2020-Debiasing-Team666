import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math
import os
import time


def get_sim_item(all_click, user_col, item_col, time_col, use_iuf=False, use_time=False):
    # 生成用户的点击商品序列，一行代表一个用户的点击序列
    user_item = all_click.groupby(user_col)[item_col].agg(list).reset_index()
    # 生成字典
    user_item_dict = dict(zip(user_item[user_col], user_item[item_col]))
    # 生成用户的点击时间序列
    user_time = all_click.groupby(user_col)[time_col].agg(list).reset_index()  # 引入时间因素
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
                        indicator = 1.0 if flag else 1.0
                        sim_item[i][j] += 1.0 * indicator * (0.8 ** num) * (1 - t) / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    for i, related_items in tqdm(sim_item.items()):
        for j, sim in related_items.items():
            # sim_item_corr[i][j] = sim / math.sqrt(item_cnt[i] * item_cnt[j])
            sim_item_corr[i][j] = sim / (item_cnt[i] * item_cnt[j]) ** 0.2

    return sim_item_corr, user_item_dict


# 交互行为打分，根据位置远近添加权重
def recommend(sim_item_corr, user_items_dict, user_id, topk, item_num):
    rank = defaultdict(list)
    if user_id not in user_items_dict:  # 如果不存在点击历史，冷启动, 是否存在用户画像，后者根据点击时间使用更短时间段范围的topk填充
        return []
    interacted_items = user_items_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:topk]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.75 ** loc)

        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]


# fill user to 50 items，召回的50个商品中，是最后一次点击的商品label是1，反之为0.
def get_predict(rec_df,user_test, pred_col, top_50_clicks):
    top_50_clicks = [int(t) for t in top_50_clicks.split(',')]
    scores = [-1 * (i + 1) for i in range(0, len(top_50_clicks))]
    ids = list(user_test)

    fill_df = pd.DataFrame(ids * len(top_50_clicks), columns=['user_id'])
    fill_df.sort_values('user_id', inplace=True)
    fill_df['item_id'] = top_50_clicks * len(ids)
    fill_df[pred_col] = scores * len(ids)
    rec_df = rec_df.append(fill_df)
    rec_df.sort_values(pred_col, ascending=False, inplace=True)
    rec_df = rec_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    rec_df['rank'] = rec_df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
    rec_df = rec_df[rec_df['rank'] <= 50]
    rec_df = rec_df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',',
                                                                                                           expand=True).reset_index()

    return rec_df


if __name__ == '__main__':
    current_phase = 6
    user_path = os.path.expanduser('~')
    train_path = os.path.join(user_path, r'kdd\data\underexpose_train')
    test_path = os.path.join(user_path, r'kdd\data\underexpose_test')
    rec_items = []
    users=set()
    whole_click = pd.DataFrame()
    for phase in range(current_phase + 1):
        print("phase: ", phase)
        train_click = pd.read_csv(train_path + '\\underexpose_train_click-{}.csv'.format(phase), header=None,
                                  names=['user_id', 'item_id', 'time'])
        test_click = pd.read_csv(test_path + '\\underexpose_test_click-{}.csv'.format(phase), header=None,
                                 names=['user_id', 'item_id', 'time'])
        # test_users = pd.read_csv(test_path + '\\underexpose_test_qtime-{}.csv'.format(phase), header=None,
        #                          names=['user_id', 'time'])

        user_test = set(test_click['user_id'])  # 每阶段线下测试集用户集合
        print('len(user_val):', len(user_test))
        users.update(user_test)

        all_click = train_click.append(test_click)
        whole_click = whole_click.append(all_click)
        whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
        whole_click = whole_click.sort_values('time')

        whole_click['pred'] = whole_click['user_id'].map(lambda x: 'test' if x in user_test else 'train')
        underline_test = whole_click[whole_click['pred'] == 'test'].drop_duplicates(['user_id'], keep='last')  # 当前阶段线下测试集click数据
        time_min = underline_test['time'].min()
        time_max = underline_test['time'].max()
        underline_train = whole_click.append(underline_test).drop_duplicates(keep=False)  # 当前阶段以及之前阶段训练集click数据
        print('len(underline_train): ', len(underline_train))
        underline_train = underline_train[(underline_train['time'] >= time_min) & (underline_train['time'] <= time_max)]
        print('len(underline_train): ', len(underline_train))

        top500_click = underline_train['item_id'].value_counts().index[:500].values  # 最热商品
        top500_click_list = list(top500_click)

        item_sim_list, user_item = get_sim_item(underline_train, 'user_id', 'item_id', 'time', use_iuf=True, use_time=True)

        for i in tqdm(user_test):  # qtime，unique()返回参数数组中所有不同的数并从小到大排序
            rank_items = recommend(item_sim_list, user_item, i, 500, 50)  # 推荐50
            for j in rank_items:
                rec_items.append([i, j[0], j[1]])

    # find most popular items for cold-start users
    top_50_clicks = whole_click['item_id'].value_counts().index[:50].values
    top_50_clicks = ','.join([str(i) for i in top_50_clicks])

    rec_df = pd.DataFrame(rec_items, columns=['user_id', 'item_id', 'sim'])
    result = get_predict(rec_df,users, 'sim', top_50_clicks)
    time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    file_name = 'underexpose_submit{time_str}.csv'.format(time_str=time_str)
    result.to_csv(file_name, index=False, header=None)


