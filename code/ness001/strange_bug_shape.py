import pandas as pd
import csv
pred_path='../../prediction_result/'
pred_df=pd.read_csv('/Users/ness001/OneDrive/推荐系统/KDD2020-Debiasing-Team666/prediction_result/recall_allvec_df.csv',low_memory=False, quoting=csv.QUOTE_NONE, error_bad_lines=False,encoding='utf8')
pred_df=pred_df.loc[pred_df['rank'] <=50 ]
preds={}
for user in pred_df.user_id.unique():
    assert len(pred_df.loc[pred_df.user_id == user ]) == 50
    print(user)
    # try:
    #     index_list=pred_df.loc[pred_df.user_id == user ].index[:50]
    #     preds[int(user)] = pred_df.iloc[index_list].item_id_pred.astype(int).to_list()
    # except IOError:
    #     print( len(pred_df.loc[pred_df.user_id == user ]) == 50, 'user id',user)
    index_list = pred_df.loc[pred_df.user_id == user].index[:50] # it returns row labels
    preds[int(user)] = pred_df.iloc[index_list].item_id_pred.astype(int).to_list()