import pandas as pd
print('pd.__version__:',pd.__version__) # 1.0.3
import numpy as np
print('np.__version__:',np.__version__) # 1.14.6
import os
user_dir=os.path.expanduser('~')
train_path = os.path.join(user_dir, r'kdd\data\underexpose_train')
file_path=os.path.join(train_path, 'underexpose_user_feat.csv')

user_feat = pd.read_csv(file_path,
                        names=["user_id", "user_age_level", "user_gender", "user_city_level"],
                        header=None, nrows=None, sep=",",
                        dtype={'user_id':np.str})

print(user_feat.head())
user_set=set(user_feat['user_id'].values)
print('len(user_set): ',len(user_set))
user_feat=user_feat