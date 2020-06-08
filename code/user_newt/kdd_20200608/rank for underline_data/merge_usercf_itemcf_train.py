import pandas as pd

for phase in range(8,10):
    print('phase=',phase)
    itemcf_file='..\\data\\train_itemcf_phase-%d.csv'
    usercf_file='..\\data\\train_usercf_phase-%d.csv'

    itemcf_df=pd.read_csv(itemcf_file % phase,
                          header=None,
                          names=['user_id','item_id','itemcf_rank','itemcf_score', 'label'])
    print('\nitemcf_df: ',itemcf_df.shape)
    print(itemcf_df.head())

    usercf_df=pd.read_csv(usercf_file % phase,
                          header=None,
                          names=['user_id','item_id','usercf_rank','usercf_score', 'label'])

    print('\nusercf_df: ',usercf_df.shape)
    print(usercf_df.head())

    df=pd.merge(itemcf_df, usercf_df, how='outer', on=['user_id','item_id','label'])


    with open('..\\data\\dict_item_users_phase-%d.txt' %phase,'r') as fout:
        temp_str=fout.read()
        dict_item_users=eval(temp_str)

    with open('..\\data\\dict_user_items_phase-%d.txt' % phase,'r') as fout:
        temp_str=fout.read()
        dict_user_items=eval(temp_str)

    df['item_heat']=df['item_id'].map(lambda x: len(dict_item_users[x]) if x in dict_item_users else 0)
    df['user_heat']=df['user_id'].map(lambda x: len(dict_user_items[x]) if x in dict_user_items else 0)

    print('\ndf: ',df.shape)
    print(df.head())
    print(df.columns.values)
    df.to_csv('..\\data\\train_merge_phase-%d.csv' % phase, header=False, index=False)