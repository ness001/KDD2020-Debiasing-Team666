1.对于文件夹rank for underline_data
  对于线下的训练集执行过程：
  recall_itemcf_train.py     # 线下训练集item召回500
  recall_usercf_train.py     # 线下训练集item召回500
  merge_usercf_itemcf_train.py   # 合并
  get_cosin_similarity_train.py      # 计算召回的item和用户购物历史的txt_vec和img_vec相似度
  recall_itemcf_test.py    #线下测试集itemcf召回
  recall_usercf_test.py     #线下测试集usercf召回
  merge_usercf_itemcf_test.py   #合并
  get_cosin_similarity_test.py   #
  train_xgboost.py           #训练xgboost做排序
  eval_itemcf_usercf_alone.py    #分别评估itemcf和usercf线下测试集指标
  eval_itemcf_usercf_merged.py   #评估xgboost排序后的线下测试集指标

  