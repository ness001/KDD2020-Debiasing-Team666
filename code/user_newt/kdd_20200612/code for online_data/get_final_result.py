"""
合并phase7,8,9的结果，并按照'score'项排序取top50 item
"""

import numpy as np
import pandas as pd

start=7
now_phase=9
submit_answer_fname='..\\test_data\\online_result_phase-%d.csv'
user_list=[]
itemlist_list=[]
for phase in range(7, 10):
    print("phase:", phase)
    df_online = pd.read_csv(submit_answer_fname % phase,
                          header=None,
                          names=['user_id', 'item_id', 'txt_sim', 'img_sim', 'itemcf_rank', 'itemcf_score',
                                 'label', 'usercf_rank', 'usercf_score', 'w2v_rank', 'w2v_score', 'item_heat',
                                 'user_heat', 'score'])
    print("df_online: ", df_online.shape)
    print(df_online.head())
    df_online = df_online.groupby('user_id')[['item_id', 'score']].agg(list)
    print("df_online: ", df_online.shape)
    print(df_online.head())
    prediction_user_set = set(df_online.index.to_list())

    for user in prediction_user_set:
        item_list = np.array(df_online.loc[user]['item_id'])
        score_array = np.array(df_online.loc[user]['score'])
        index = np.argsort(score_array)[::-1]
        item_list = list(item_list[index])
        score_list = list(score_array[index])
        user_list+=([user]*50)
        itemlist_list+=item_list[:50]

temp_dict = dict()
temp_dict['user_id'] = user_list
temp_dict['item_id'] = itemlist_list
df = pd.DataFrame(data=temp_dict)
print('df: ',df.shape)
print(df.head())
df = df.groupby('user_id')['item_id'].\
        apply(lambda x: ','.join([str(i) for i in x])).str.split(',',expand=True).reset_index()

print('df: ',df.shape)
print(df.head())

df.to_csv('..\\online_data\\underexpose_submit-9.csv',header=False,index=False)

