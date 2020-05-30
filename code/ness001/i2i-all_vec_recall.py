#!/usr/bin/env python
# coding: utf-8

# In[15]:


import sys

print(sys.version, sys.platform, sys.executable)
from tqdm import tqdm

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.precision',16)

train_path = '../../data/underexpose_train'
test_path = '../../data/underexpose_test'
user_path='../../user_data/'
# import ast
# results= ast.literal_eval(open(user_path+'result_vec.txt').read())
import json

results= open(user_path+'result_vec.txt').read()
json_acceptable_string = results.replace("'", "\"")

results=json.loads(json_acceptable_string)
p = 6
user_last = pd.read_csv(test_path + 'underexpose_test_click-{}.csv'.format(p),names=['user_id','item_id','time'])
user_last=user_last.sort_values(['user_id', 'time'],ascending=[True,False]).reset_index(drop=True)
user_true = user_last.drop_duplicates(['user_id'], keep='first')


user_train = user_last.iloc[user_true.index+1,:]


# In[ ]:




# In[19]:


recall_num=450
itemlist=list(itemft.item_id.unique())
recall_frame = pd.DataFrame(columns=['user_id', 'item_id_pred', 'score', 'rank', 'item_id_true'],index=range(0,recall_num*len(user_train.user_id.unique())))
row_num=0
for user in tqdm(user_train.user_id.unique()):
    latest_item = user_train.loc[user_train.user_id == user, 'item_id'].values[0]
    true_item = user_true.loc[user_true.user_id == user, 'item_id'].values[0]
    if latest_item in itemlist: 
        recs = results[latest_item][:recall_num]
        for i in range(0, recall_num):
            
            recall_frame.iloc[row_num]['user_id'] = user
            recall_frame.iloc[row_num]['item_id_pred'] = recs[i][1]
            recall_frame.iloc[row_num]['score'] = recs[i][0]
            recall_frame.iloc[row_num]['rank'] = i+1
            recall_frame.iloc[row_num]['item_id_true'] = true_item
            row_num+=1
    else:
#         print(user,latest_item,latest_item in itemlist)
        recall_frame.iloc[row_num]['user_id'] = user
        recall_frame.iloc[row_num]['item_id_pred'] = np.nan
        recall_frame.iloc[row_num]['score'] = np.nan
        recall_frame.iloc[row_num]['rank'] = np.nan
        recall_frame.iloc[row_num]['item_id_true'] = true_item
        row_num+=1
recall_frame

recall_frame=recall_frame.dropna(how='all').astype('float32')
sum(recall_frame.item_id_pred == recall_frame.item_id_true)/1821


# In[ ]:


recall_frame.to_csv('recall_allvec_df.csv',index=False)


# In[ ]:


# import faiss

# vec=np.ascontiguousarray(vec.astype('float32'))

# dim=vec.shape[1]
# k=500 # recall 
# nlist=100
# quantizer = faiss.IndexFlatL2(dim)
# idx=faiss.IndexIVFPQ(quantizer,dim,nlist,10,8)
# idx.nprobe =10

# idx.train(vec)
# idx.add(vec)
# d,i=idx.search(vec,100)


# In[ ]:




