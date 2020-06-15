import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import math
df=pd.read_csv('/Users/ness001/OneDrive/推荐系统/KDD2020-Debiasing-Team666/data/underexpose_train/underexpose_train_click-0.csv',names=['user_id','item_id','time'],low_memory=False)

user_col='user_id'
item_col='item_id'
user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()
user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()
item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

sim_item = {}

for item, users in tqdm(item_user_dict.items()):

    sim_item.setdefault(item, {})

    for u in users:

        tmp_len = len(user_item_dict[u])

        for relate_item in user_item_dict[u]:
            sim_item[item].setdefault(relate_item, 0)
            sim_item[item][relate_item] += 1 / (math.log(len(users) + 1) * math.log(tmp_len + 1))
a=1
