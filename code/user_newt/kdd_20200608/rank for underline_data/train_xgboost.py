from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
'''
phase-%d
underline_train for get features and underline_val for train
underline_test for eval to get top50 item
'''

train_file='..\\data\\train_feat_phase-%d.csv'
test_file='..\\test_data\\test_feat_phase-%d.csv'
start=9
end=10
ratio=50    # 正负样本比例

for phase in range(start, end):
    print('phase: ', phase)
    print('------------train-----------')
    df_train=pd.read_csv(train_file % phase,
                        header=None,
                        names=['user_id','item_id','txt_sim', 'img_sim', 'itemcf_rank','itemcf_score',
                               'label', 'usercf_rank', 'usercf_score', 'item_heat', 'user_heat'])

    print('df_train: ',df_train.shape)
    print(df_train.head())

    pos_samples=df_train[df_train['label']==1]
    print('len(pos_samples):', len(pos_samples))
    neg_samples=df_train[df_train['label']==0]
    neg_samples=shuffle(neg_samples)[:50*len(pos_samples)]
    print('len(neg_samples):', len(neg_samples))

    samples=pos_samples.append(neg_samples)
    samples=shuffle(samples)
    Y_train=samples['label']
    X_train=samples.drop(labels=['user_id','item_id','label'],axis=1)
    print(X_train.shape)
    # print(X_train[:10].values)
    # print(Y_train[:10].values)

    # step1 调试n_estimators
    # cv_params = {'n_estimators': [100, 300, 500, 700]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 10, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    #step2 调试max_depth，min_child_weight
    cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 7, 'min_child_weight': 2, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}


    model = XGBClassifier(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='recall', cv=3, verbose=1, n_jobs=4)
    # optimized_GBM.fit(X_train, Y_train)
    #
    # evalute_result = optimized_GBM.cv_results_
    #
    # print('\n每轮迭代运行结果:')
    # for key,value in evalute_result.items():
    #     print(key,value)
    #
    # print('\n参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    #
    # print('\n最佳模型得分:{0}'.format(optimized_GBM.best_score_))


    model.fit(X_train.values, Y_train.values)

    prob_pred=model.predict_proba(X_train.values)

    print(prob_pred[:10])

    y_pred=prob_pred.argsort(axis=1)[:, 1]

    print(y_pred[:10])

    acc=accuracy_score(Y_train, y_pred)
    print('acc:',acc)

    p=precision_score(Y_train,y_pred)
    print('p:', p)

    recall=recall_score(Y_train, y_pred)
    print('recall:',recall)

    f1=f1_score(Y_train,y_pred)
    print('f1:',f1)


    print('------------eval-----------')
    test_df=pd.read_csv( test_file % phase,
                        header=None,
                        names=['user_id', 'item_id', 'txt_sim', 'img_sim', 'itemcf_rank', 'itemcf_score',
                               'label', 'usercf_rank', 'usercf_score', 'item_heat', 'user_heat'])
    print('test_df:')
    print(test_df.shape)
    print(test_df.head())

    Y_test=test_df['label'].values
    # print(Y_test[:10])
    X_test=test_df.drop(labels=['user_id','item_id','label'],axis=1).values
    print('X_test.shape:', X_test.shape)
    print('Y_test.shape:', Y_test.shape)

    prob_pred=model.predict_proba(X_test)
    print('prob_pred.shape: ', prob_pred.shape)

    y_pred=prob_pred.argsort(axis=1)[:,1]

    test_acc=accuracy_score(Y_test,y_pred)
    print('test_acc: ', test_acc)

    p=precision_score(Y_test,y_pred)
    print('test_p:', p)

    recall=recall_score(Y_test,y_pred)
    print('test_recall:',recall)

    f1=f1_score(Y_test,y_pred)
    print('test_f1:',f1)

    test_df['score']=prob_pred[:,1]
    print('test_df.shape:', test_df.shape)
    print(test_df.head(20))
    test_df.to_csv('..\\test_data\\test_result_phase-%d.csv' % phase,header=False,index=False)

    # joblib.dump(model,  "..\\data\\xgboost_model.pkl")



