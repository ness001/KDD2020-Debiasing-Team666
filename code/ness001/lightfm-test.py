# !pip install pandas==0.21
# !pip install tqdm 
# !pip install lightfm

# !pip install loguru

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from loguru import logger
import argparse
import multiprocessing






parser = argparse.ArgumentParser(description='t')
parser.add_argument('--now_phase', type=int, default=6, help='')
parser.add_argument('--window', type=int, default=10, help='cocur_matr的时间窗')
parser.add_argument('--time_decay', type=float, default=7/8, help='时间衰减')
parser.add_argument('--mode', type=str, default='train', help='train test')
parser.add_argument('--topk', type=int, default=500, help='每种召回策略召回的样本数')
parser.add_argument('--DATA_DIR', type=str, default='./', help='data dir')

args = parser.parse_args(args=[])
trace = logger.add(os.path.join(args.DATA_DIR, 'data_gen/runtime.log'))

# Cell
def load_click_data_per_phase(now_phase, base_dir):
    """
    """


    all_click_df = []
    for c in range(now_phase + 1):
        logger.info(f'phase: {c}')
        cols_str = 'user_id item_id time'.split()
        click_train1 = pd.read_csv( './underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})
        click_test1 = pd.read_csv( './underexpose_test_click-{}.csv'.format(c, c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})
        test_qtime1 = pd.read_csv( './underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','time'], converters={c: str for c in cols_str})
        click_test1_val = click_test1.sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id'],keep='last')

        click_test1 = click_test1[~click_test1.index.isin(click_test1_val.index)]
        all_click = click_train1.append(click_test1).drop_duplicates().sort_values('time')

        all_click_df.append((all_click, click_test1_val, test_qtime1))
        logger.info(f'all_click: {all_click.shape}, click_test1_val: {click_test1_val.shape}, test_qtime1: {test_qtime1.shape}')
    return all_click_df
all_click_df = load_click_data_per_phase(args.now_phase, args.DATA_DIR)
all_train=pd.concat([all_click_df[i][0] for i in range(0,7)]).drop_duplicates().reset_index(drop=True)
all_train=all_train.sort_values(by=['user_id','time'],ascending=['True','False'])


col_name=['item_id']
for i in range(0,128):
    col_name.append('tv'+str(i))
for i in range(0,128):
    col_name.append('iv'+str(i))
itemft=pd.read_csv('./underexpose_item_feat.csv',low_memory=False,names=col_name)
itemft=itemft.replace('[\[\]]','',regex=True)#regex=True is the key
itemft=itemft.astype({'item_id':int, 'tv0':float, 'tv127':float, 'iv0':float, 'iv127':float})


userft=pd.read_csv('./underexpose_user_feat.csv',low_memory=False,names=['user_id','age','gender','city'],dtype=\
                  {'user_id': int,'age':'object','gender':'object','city':'object'})

# user feat have duplicated user ids
a=userft.user_id.value_counts()
dup_user_id=a[a>1].index.tolist()
dup_index=userft.loc[userft.user_id.isin(dup_user_id),:].index.tolist()
dup_index=[i for i in dup_index if i%2 == 0]
userft.drop(dup_index,inplace=True)

all_train=all_click_df[6][0].astype(float)
all_test=all_click_df[6][1].astype(float)
all_test=all_test[(all_test['user_id'].isin(all_train['user_id'])) & (all_test['item_id'].isin(all_train['item_id']))]

from sklearn.preprocessing import LabelEncoder
le_cols=['item_id','user_id']
encoded_train=dict()
encoded_test=dict()
for key in le_cols:
    le=LabelEncoder()
    encoded_train[key]=le.fit_transform(all_train[key].values)
    encoded_test[key]=le.transform(all_test[key].values)


n_users,n_items=len(np.unique(encoded_train['user_id'])),len(np.unique(encoded_train['item_id']))

n_users,n_items

train_codata=np.ones(shape=(all_train.shape[0],))
test_codata=np.ones(shape=(all_test.shape[0],))

from scipy.sparse import coo_matrix

train = coo_matrix((train_codata,(encoded_train['user_id'],encoded_train['item_id'])),shape=(n_users,n_items))
test = coo_matrix((test_codata,(encoded_test['user_id'],encoded_test['item_id'])),shape=(n_users,n_items))

user_item_matrix=all_train.pivot_table(index='user_id',columns='item_id',aggfunc=len,fill_value=0)
users=pd.DataFrame(data=list(user_item_matrix.index),columns=['user_id'])
user_features=users.merge(userft,how='left')
user_features.shape[0] == user_item_matrix.shape[0]


items=pd.DataFrame(data=list(user_item_matrix.columns.droplevel(level=0)),columns=['item_id'])
item_features=items.merge(itemft,how='left')
item_features.shape[0] == user_item_matrix.shape[1]


user_features.fillna(0,inplace=True)
item_features.fillna(0,inplace=True)

user_features.info()
item_features.info()

user_features=user_features.replace({'M':1,'F':0})

user_features=user_features.astype({'user_id':int,'age':int,'gender':int,'city':int})

user_features.set_index('user_id',inplace=True)
item_features.set_index('item_id',inplace=True)

user_features.dropna().shape
user_features.shape
item_features.dropna().shape
item_features.shape


from scipy.sparse import csr_matrix
uft_csr=csr_matrix(user_features.values)
ift_csr=csr_matrix(item_features.values)
uft_csr.shape
ift_csr.shape


model=LightFM(learning_rate=0.05,loss='bpr')
model.fit(train,epochs=30,item_features=ift_csr,verbose=True,num_threads=num_threads-1)

def fm_eval(coo,prec_k=10,recall_k=100,use='itemft',train_interactions=None,ift_csr=None,t=1):
    from lightfm.evaluation import precision_at_k,auc_score,recall_at_k
    if use == 'itemft':
        train_precision = precision_at_k(model,coo,train_interactions=train_interactions,item_features=ift_csr,k=prec_k,num_threads=t).mean()
        train_auc = auc_score(model,coo,train_interactions=train_interactions,item_features=ift_csr,num_threads=t).mean()
        train_recall=recall_at_k(model,coo,train_interactions=train_interactions,k=recall_k,item_features=ift_csr,num_threads=t).mean()
        print('train: prec: {} with top {},auc: {} , recall: {} with top {}'.format(train_precision,prec_k,train_auc,train_recall,recall_k))
    if use == None:
        train_precision = precision_at_k(model,coo,train_interactions=train_interactions,k=prec_k,num_threads=t).mean()
        train_auc = auc_score(model,coo,train_interactions=train_interactions,num_threads=t).mean()
        train_recall=recall_at_k(model,coo,train_interactions=train_interactions,k=recall_k,num_threads=t).mean()
        print('train: prec: {} with top {},auc: {} , recall: {} with top {}'.format(train_precision,prec_k,train_auc,train_recall,recall_k))
fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use='itemft',ift_csr=ift_csr,t=num_threads-1)
model.item_biases *= 0.0
print('with 0 bias!')
fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use='itemft',ift_csr=ift_csr,t=num_threads-1)

