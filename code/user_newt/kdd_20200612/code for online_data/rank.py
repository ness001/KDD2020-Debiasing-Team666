from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier
import joblib
from xgboost.sklearn import XGBClassifier
'''
phase-%d
underline_train for get features and underline_val for train
underline_test for eval to get top50 item
'''

start_phase=7
now_phase=9
online_file='..\\online_data\\online_feat_phase-%d.csv'
ratio=50    # 正负样本比例

for phase in range(start_phase, now_phase+1):
    model=joblib.load("..\\test_data\\xgboost_model_2_phase_%d.pkl" % phase)
    print('------------eval-----------')
    df_online=pd.read_csv( online_file % phase,
                        header=None,
                        names=['user_id', 'item_id', 'txt_sim', 'img_sim', 'itemcf_rank', 'itemcf_score',
                               'label', 'usercf_rank', 'usercf_score', 'w2v_rank', 'w2v_score', 'item_heat', 'user_heat'])
    print('df_online:')
    print(df_online.shape)
    print(df_online.head())

    X_test=df_online.drop(labels=['user_id','item_id','label'],axis=1).values
    print('X_test.shape:', X_test.shape)

    prob_pred=model.predict_proba(X_test)
    print('prob_pred.shape: ', prob_pred.shape)

    df_online['score']=prob_pred[:,1]

    print('df_online: ', df_online.shape)
    print(df_online.head(20))
    df_online.to_csv('..\\online_data\\online_result_phase-%d.csv' % phase,header=False,index=False)





