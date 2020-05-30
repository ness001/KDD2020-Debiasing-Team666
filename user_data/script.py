now_phase=6
user_path='../../'
for c in range(0,now_phase+1):
    test_path = '../data/underexpose_test/'  
    df=pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c,c), header=None,  names=['user_id', 'item_id', 'time'])
#  the same outcome   
#df.sort_values(by=['user_id','time'],ascending=False).groupby(['user_id']).agg('max').reset_index()
    temp=df.sort_values(by=['user_id','time'],ascending=False).drop_duplicates(subset=['user_id'],keep='first')
    temp.to_csv(test_path+'utc_click-{}.csv'.format(c),index=False,header=False)
