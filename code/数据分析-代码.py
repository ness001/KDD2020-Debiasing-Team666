
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# # 读取数据

# ## 候选集——用户候选集，商品候选集

# ## 用户候选集

# In[4]:


underexpose_user_feat = pd.read_csv("../data/underexpose_train/underexpose_user_feat.csv"
                                    ,names=["user_id", "user_age_level", "user_gender", "user_city_level"]
                                    ,nrows=None      
                                    ,sep=","
                                   )


# In[5]:




underexpose_user_feat.user_age_level.value_counts(dropna=False,ascending=False)


# ## 查看用户user_gender分布

# In[8]:


underexpose_user_feat.user_gender.value_counts(dropna=False,ascending=False)


# ## 查看用户user_city_level分布

# In[9]:


underexpose_user_feat.user_city_level.value_counts(dropna=False,ascending=False)


# ## 商品候选集

# In[10]:


col_name=['id']
for i in range(0,128):
    col_name.append('tv'+str(i))
for i in range(0,128):
    col_name.append('iv'+str(i))


# In[11]:


underexpose_item_feat = pd.read_csv("../data/underexpose_train/underexpose_item_feat.csv",low_memory=False,names=col_name)

underexpose_item_feat.head(3).dtypes
# In[12]:


underexpose_item_feat.head(3)['tv0'].str.replace('[','')


# In[13]:


underexpose_item_feat.head(3).apply(lambda x: x.str.replace('[\[\]]',''),axis=1)## mixed dtypes gen many nans


# In[14]:


underexpose_item_feat[underexpose_item_feat.select_dtypes('object').columns]=underexpose_item_feat.select_dtypes('object').apply(lambda x: x.str.replace('[\[\]]',''),axis=1)


# In[15]:


underexpose_item_feat


# # 训练集——用户点击商品数据

# In[31]:


list_phase_range = [0,1,2,3,4]
list_train_phase_file = [(x_,"../data/underexpose_train/underexpose_train_click-"+str(x_)+".csv") for x_ in list_phase_range]


# In[32]:


list_train_phase_file


# In[33]:


def get_train_click(list_train_phase_file):
    list_underexpose_click = []
    columns = ["user_id", "item_id", "time", "phase"]
    for phase, file in tqdm(list_train_phase_file):
        underexpose_click = pd.read_csv(file
                                        ,names=["user_id", "item_id", "time"]
                                        ,nrows=None      
                                        ,sep=","
                                       )
        underexpose_click["phase"] = phase
        underexpose_click = underexpose_click[columns]
        list_underexpose_click.append(underexpose_click)
    all_underexpose_click = pd.concat(list_underexpose_click)
    return all_underexpose_click[columns]


# In[34]:


underexpose_train_click = get_train_click(list_train_phase_file)


# ## 查看训练集--点击用户点击商品数据

# In[35]:


underexpose_train_click.sort_values(['user_id','time']).head(20)


# In[36]:


underexpose_train_click[underexpose_train_click['item_id']==-999]


# In[37]:


underexpose_train_click[underexpose_train_click['user_id']==-999]


# # 测试集——用户点击数据-待预测数据

# In[38]:


list_phase_range = [0,1,2,3,4]
list_test_phase_file = [(x_,"../data/underexpose_test/underexpose_test_click-"+str(x_)+"/underexpose_test_click-"+str(x_)+".csv") for x_ in list_phase_range]
list_test_phase_query_file = [(x_,"../data/underexpose_test/underexpose_test_click-"+str(x_)+"/underexpose_test_qtime-"+str(x_)+".csv") for x_ in list_phase_range]


# In[39]:


def get_test_click(list_test_phase_file, list_test_phase_query_file):
    list_underexpose_click = []
    columns = ["user_id", "item_id", "time", "phase"]
    for phase, file in tqdm(list_test_phase_file):
        underexpose_click = pd.read_csv(file
                                        ,names=["user_id", "item_id", "time"]
                                        ,nrows=None      
                                        ,sep=","
                                       )
        underexpose_click["phase"] = phase
        underexpose_click = underexpose_click[columns] 
        list_underexpose_click.append(underexpose_click)
    for phase, file in tqdm(list_test_phase_query_file):
        underexpose_qtime = pd.read_csv(file
                                    ,names=["user_id", "query_time"]
                                    ,nrows=None      
                                    ,sep=","
                                   )   
        underexpose_qtime.columns = ["user_id", "time"]
        underexpose_qtime["phase"] = phase
        underexpose_qtime["item_id"] = -999
        underexpose_qtime = underexpose_qtime[columns]
        list_underexpose_click.append(underexpose_qtime)
    all_underexpose_click = pd.concat(list_underexpose_click)
    return all_underexpose_click[columns]


# In[40]:


underexpose_test_click = get_test_click(list_test_phase_file, list_test_phase_query_file)


# ## 查看测试集合数据-待预测的数据

# In[41]:


len(underexpose_test_click[underexpose_test_click["item_id"]== -999])


# In[42]:


len(underexpose_test_click[underexpose_test_click["item_id"]!=-999])


# # 建模数据准备

# ## 序列数据处理

# In[43]:


def deal_click_data(underexpose_click_data):
    underexpose_click_data = underexpose_click_data.sort_values(['user_id','phase','time'])
    dict_user_phase_action = {}
    for i,row in tqdm(underexpose_click_data.iterrows()):
        user_id, item_id, time, phase = int(row["user_id"]), int(row["item_id"]), float(row["time"]), int(row["phase"])
        if phase not in dict_user_phase_action:
            dict_user_phase_action[phase] = {}
        if user_id not in dict_user_phase_action[phase]:
            dict_user_phase_action[phase][user_id] = {"item_seq":[],"time_seq":[],"diff_time_seq":[]}
        else:
            diff_time = (time - dict_user_phase_action[phase][user_id]["time_seq"][-1]) * 10**4
            dict_user_phase_action[phase][user_id]["diff_time_seq"].append(diff_time)    
        dict_user_phase_action[phase][user_id]["item_seq"].append(item_id)
        dict_user_phase_action[phase][user_id]["time_seq"].append(time)
    return dict_user_phase_action


# In[44]:


# 2 * 10 **4


# In[45]:


dict_train_user_phase_action = deal_click_data(underexpose_train_click)


# In[46]:


dict_test_user_phase_action = deal_click_data(underexpose_test_click)


# ## 查看处理好的序列数据

# ### 训练数据

# In[47]:


dict_train_user_phase_action.keys()


# In[49]:


dict_train_user_phase_action[0]


# In[ ]:


dict_train_user_phase_action[0][1]


# ### 测试数据

# In[ ]:


dict_test_user_phase_action.keys()


# In[ ]:


dict_test_user_phase_action[0].keys()


# In[ ]:


dict_test_user_phase_action[0][11]


# ## 用户画像数据处理

# In[ ]:


underexpose_user_feat.head()


# In[ ]:


def deal_user_feat_data(user_feat_data):
    user_feat_data = user_feat_data.fillna(-1)
    dict_user_feat = {}
    dict_user_age_level, index_user_age_level = {}, 0
    dict_user_gender, index_user_gender = {}, 0
    dict_user_city_level, index_user_city_level = {}, 0
    for i,row in tqdm(user_feat_data.iterrows()):
        user_id, user_age_level, user_gender, user_city_level = int(row["user_id"]), int(row["user_age_level"]), row["user_gender"], int(row["user_city_level"])
        if user_id not in dict_user_feat:
            dict_user_feat[user_id] = {}
        if user_age_level not in dict_user_age_level:
            dict_user_age_level[user_age_level] = index_user_age_level
            index_user_age_level += 1
        if user_gender not in dict_user_gender:
            dict_user_gender[user_gender] = index_user_gender
            index_user_gender += 1
        if user_city_level not in dict_user_city_level:
            dict_user_city_level[user_city_level] = index_user_city_level
            index_user_city_level += 1
        dict_user_feat[user_id]["user_age_level"] = dict_user_age_level[user_age_level]
        dict_user_feat[user_id]["user_gender"] = dict_user_gender[user_gender]
        dict_user_feat[user_id]["user_city_level"] = dict_user_city_level[user_city_level]
    return dict_user_feat, dict_user_age_level, dict_user_gender, dict_user_city_level


# In[ ]:


dict_user_feat, dict_user_age_level, dict_user_gender, dict_user_city_level = deal_user_feat_data(underexpose_user_feat)


# In[ ]:


dict_user_feat[17]


# In[ ]:


dict_user_age_level


# In[ ]:


dict_user_gender


# In[ ]:


dict_user_city_level


# ## 商品画像数据处理

# In[ ]:


list_item_id = []
list_txt_vec = []
list_img_vec = []
with open("../data/underexpose_train/underexpose_item_feat.csv") as f:
    for line in tqdm(f):
        line_split = line.strip().split(',[')
        list_item_id.append(line_split[0])
        list_txt_vec.append(line_split[1].strip(']'))
        list_img_vec.append(line_split[2].strip(']'))
underexpose_item_feat = pd.DataFrame({"item_id":list_item_id,"txt_vec":list_txt_vec,"img_vec":list_img_vec})
underexpose_item_feat = underexpose_item_feat[["item_id", "txt_vec", "img_vec"]]


# In[ ]:


underexpose_item_feat.head()


# In[ ]:


def deal_item_feat_data(item_feat_data):
    dict_item_feat = {}
    for i,row in tqdm(item_feat_data.iterrows()):    
        item_id, txt_vec, img_vec = int(row["item_id"]), [float(x_) for x_ in row["txt_vec"].split(",")], [float(x_) for x_ in row["img_vec"].split(",")]   
        if item_id not in dict_item_feat:
            dict_item_feat[item_id] = {}
        dict_item_feat[item_id]["txt_vec"] = txt_vec
        dict_item_feat[item_id]["img_vec"] = img_vec
    index_user_idreturn dict_item_feat


# In[ ]:


dict_item_feat = deal_item_feat_data(underexpose_item_feat)


# In[ ]:


dict_item_feat[1]


# In[ ]:





# # id 转化为embeeding特征

# ## user_id, item_id 特征

# In[ ]:


def get_node_user_item(underexpose_train_click, underexpose_test_click):
    list_user_id = []
    list_item_id = []
    list_index_user_id = []
    list_index_item_id = []
    list_score = []
    index_user_id = 0
    index_item_id = 0
    dict_user_id = {}
    dict_item_id = {}
    list_underexpose_click = [underexpose_train_click, underexpose_test_click]
    for underexpose_click in list_underexpose_click:
        for i,row in tqdm(underexpose_click.iterrows()):
            user_id, item_id, time, phase = int(row["user_id"]), int(row["item_id"]), float(row["time"]), int(row["phase"])
            if item_id > 0:
                list_user_id.append(user_id)
                list_item_id.append(item_id)
                list_score.append(1)
                if user_id not in dict_user_id:
                    dict_user_id[user_id] = index_user_id
                    index_user_id += 1
                list_index_user_id.append(dict_user_id[user_id])
                if item_id not in dict_item_id:
                    dict_item_id[item_id] = index_item_id
                    index_item_id += 1
                list_index_item_id.append(dict_item_id[item_id])
    node_user_item = pd.DataFrame({"user_id":list_user_id, "item_id":list_item_id, "score":list_score, "index_user_id":list_index_user_id, "index_item_id":list_index_item_id})
    node_user_item = node_user_item.groupby(["user_id", "item_id", "index_user_id", "index_item_id"])["score"].sum().reset_index()
    node_user_item.columns = ["user_id", "item_id", "index_user_id", "index_item_id", "score"]
    return node_user_item, dict_user_id, dict_item_id


# In[ ]:


node_user_item, dict_user_id, dict_item_id = get_node_user_item(underexpose_train_click, underexpose_test_click)


# ### 探索数据

# In[ ]:


node_user_item.index_user_id.min()


# In[ ]:


node_user_item.index_user_id.max()


# In[ ]:


node_user_item.index_item_id.min()


# In[ ]:


node_user_item.index_item_id.max()


# ### underexpose_user_feat 用户画像只覆盖部分用户

# 训练集中有的用户 和 用户画像 共有

# In[ ]:


len((set(underexpose_test_click.user_id.astype(str)) |
    set(underexpose_train_click.user_id.astype(str)))
    & set(underexpose_user_feat.user_id.astype(str)))


# 训练集中有的用户 有 和 用户画像 没有

# In[ ]:


len((set(underexpose_test_click.user_id.astype(str)) |
    set(underexpose_train_click.user_id.astype(str))) 
    - set(underexpose_user_feat.user_id.astype(str)))


# 训练集中有的用户 没有 和 用户画像 有

# In[ ]:


len(set(underexpose_user_feat.user_id.astype(str)) 
    - (set(underexpose_test_click.user_id.astype(str)) |
    set(underexpose_train_click.user_id.astype(str))))


# ### underexpose_item_feat 商品画像只覆盖部分商品

# 训练集中有的商品 和 商品画像 共有

# In[ ]:


len((set(underexpose_test_click.item_id.astype(str)) |
    set(underexpose_train_click.item_id.astype(str)))
    & set(underexpose_item_feat.item_id.astype(str)))


# 训练集中有的商品 有 和 商品画像 没有

# In[ ]:


len((set(underexpose_test_click.item_id.astype(str)) |
    set(underexpose_train_click.item_id.astype(str))) 
    - set(underexpose_item_feat.item_id.astype(str)))


# 训练集中有的商品 没有 和 商品画像 有

# In[ ]:


len(set(underexpose_item_feat.item_id.astype(str)) 
    - (set(underexpose_test_click.item_id.astype(str)) |
    set(underexpose_train_click.item_id.astype(str))))


# # 导入tensorflow

# In[ ]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# ## 矩阵分解 图算法 Word2Vec FM

# ## 矩阵分解

# In[ ]:


class Decompose:
    def __init__(self
                 ,size_user_id
                 ,size_item_id
                 ,embedding_size
                 ,learning_rate
                ):
        self.size_user_id = size_user_id+1
        self.size_item_id = size_item_id+1
        self.embedding_size = 128
        self.learning_rate = learning_rate
        self.model()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()

    def model(self):
        self.input_user_id = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_item_id = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_score = tf.placeholder(dtype=tf.float32, shape=[None])

        self.variable_user_id = tf.Variable(tf.random_uniform([self.size_user_id, self.embedding_size], -1.0, 1.0), name="variable_user_id")
        self.variable_item_id = tf.Variable(tf.random_uniform([self.size_item_id, self.embedding_size], -1.0, 1.0), name="variable_item_id")

        self.embeeding_user_id = tf.nn.embedding_lookup(self.variable_user_id, self.input_user_id)
        self.embeeding_item_id = tf.nn.embedding_lookup(self.variable_item_id, self.input_item_id)

        self.output_score = tf.reduce_sum(self.embeeding_user_id * self.embeeding_item_id, axis=1)
        self.loss = tf.reduce_mean((self.input_score-self.output_score) ** 2)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def initializer(self):
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, feed_dict):
        return self.sess.run([self.loss, self.optimizer], feed_dict)
        
    def loss(self, feed_dict):
        return self.sess.run(self.loss, feed_dict)

    def get_embeeding_user_id(self, feed_dict):
        return self.sess.run(self.embeeding_user_id, feed_dict)
        
    def get_embeeding_item_id(self, feed_dict):
        return self.sess.run(self.embeeding_item_id, feed_dict)
        
    def save(self, model_name):
        self.saver.save(self.sess, "./tf_model/"+model_name)
    
    def load(self, model_name):
        self.saver.restore(self.sess, "./tf_model/"+model_name)


# In[ ]:


size_user_id = node_user_item.index_user_id.max()
size_item_id = node_user_item.index_item_id.max()
embedding_size = 128
learning_rate = 0.03


# In[ ]:


decompose = Decompose(
                  size_user_id=size_user_id
                 ,size_item_id=size_item_id
                 ,embedding_size=embedding_size
                 ,learning_rate=learning_rate
                )
decompose.initializer()


# In[ ]:


decompose.load("decompose.model05")


# In[ ]:


node_user_item.head()


# In[ ]:


set_index_user_id = set(node_user_item["index_user_id"])
set_index_item_id = set(node_user_item["index_item_id"])
# set_index_user_id_item_id = set([str(index_user_id)+"-"+str(index_item_id) for (index_user_id, index_item_id) in zip(node_user_item["index_user_id"], node_user_item["index_item_id"])])
set_index_user_id_index_item_id = set(zip(node_user_item["index_user_id"], node_user_item["index_item_id"]))


# In[ ]:


# (17811, 732) in set_index_user_id_item_id


# In[ ]:


import random


# In[ ]:


size_item_id


# In[ ]:


size_user_id


# In[ ]:


random.sample(set_index_item_id, 5)


# In[ ]:


sample_num = 5
epoch_num = 10


# In[ ]:


len(set_index_user_id_index_item_id)


# In[ ]:


index = 0
n = 1000
# max_index = 1000


# In[ ]:


log_file = open('./temp/log/decompose.log','w')


# In[ ]:


for epoch in tqdm(range(epoch_num)):
    for (index_user_id, index_item_id) in set_index_user_id_index_item_id:
        list_input_user_id = []
        list_input_item_id = []
        list_input_score = []

        list_input_user_id.append(index_user_id)
        list_input_item_id.append(index_item_id)
        list_input_score.append(1)

        for sample_index_item_id in random.sample(set_index_item_id, sample_num):
            if (index_item_id, sample_index_item_id) not in set_index_user_id_index_item_id:
                list_input_user_id.append(index_user_id)
                list_input_item_id.append(sample_index_item_id)
                list_input_score.append(0)

        for sample_index_user_id in random.sample(set_index_user_id, sample_num):
            if (sample_index_user_id, index_item_id) not in set_index_user_id_index_item_id:
                list_input_user_id.append(sample_index_user_id)
                list_input_item_id.append(index_item_id)
                list_input_score.append(0)   

        feed_dict = {
            decompose.input_user_id:list_input_user_id,
            decompose.input_item_id:list_input_item_id,
            decompose.input_score:list_input_score
        }
        loss, optimizer = decompose.train(feed_dict)
        index += 1
        if index % n == 0:
            log_file.write("epoch : " + str(epoch) +", index : " + str(index) + ", loss :" + str(loss) + "\n")
            log_file.flush()
# decompose.save("decompose.model")


# In[ ]:





# In[ ]:


for epoch in tqdm(range(1)):
    for (index_user_id, index_item_id) in random.sample(set_index_user_id_index_item_id, 20):
        list_input_user_id = []
        list_input_item_id = []
        list_input_score = []

#         list_input_user_id.append(index_user_id)
#         list_input_item_id.append(index_item_id)
#         list_input_score.append(1)

        for sample_index_item_id in random.sample(set_index_item_id, sample_num):
            if (index_item_id, sample_index_item_id) not in set_index_user_id_index_item_id:
                list_input_user_id.append(index_user_id)
                list_input_item_id.append(sample_index_item_id)
                list_input_score.append(0)

        for sample_index_user_id in random.sample(set_index_user_id, sample_num):
            if (sample_index_user_id, index_item_id) not in set_index_user_id_index_item_id:
                list_input_user_id.append(sample_index_user_id)
                list_input_item_id.append(index_item_id)
                list_input_score.append(0)   

        feed_dict = {
            decompose.input_user_id:list_input_user_id,
            decompose.input_item_id:list_input_item_id,
            decompose.input_score:list_input_score
        }
        print(list_input_score)
        loss, optimizer = decompose.train(feed_dict)
        print(loss)
        
        list_input_user_id = []
        list_input_item_id = []
        list_input_score = []
        
        list_input_user_id.append(index_user_id)
        list_input_item_id.append(index_item_id)
        list_input_score.append(1)
        
        feed_dict = {
            decompose.input_user_id:list_input_user_id,
            decompose.input_item_id:list_input_item_id,
            decompose.input_score:list_input_score
        }
        print(list_input_score)
        loss, optimizer = decompose.train(feed_dict)
        print(loss)  

#         index += 1
        
#         if index > 20:
#             break
#         if index % n == 0:
#             log_file.write("epoch : " + str(epoch) +", index : " + str(index) + ", loss :" + str(loss) + "\n")
#             log_file.flush()
#         break


# In[ ]:





# In[ ]:





# In[ ]:


# decompose.save("decompose.model05")


# In[ ]:


# decompose.save("decompose.model")


# In[ ]:


# user_id	item_id	index_user_id	index_item_id	score


# ## 抽取user_id的embeeding

# In[ ]:


dict_index_user_id_to_user_id = dict(zip(node_user_item.index_user_id,node_user_item.user_id))
dict_user_id_embeeding = {}
for index_user_id in set_index_user_id:
    user_id_embeeding = decompose.get_embeeding_user_id(feed_dict={decompose.input_user_id:[index_user_id]})
    dict_user_id_embeeding[dict_index_user_id_to_user_id[index_user_id]] = user_id_embeeding[0]


# In[ ]:


def save_embeeding(dict_embeeding, file_name):
    with open(file_name,'w') as f:
        for k in tqdm(dict_embeeding.keys()):
            embeeding = ','.join([str(x_) for x_ in dict_embeeding[k]])
            f.write(str(k)+','+embeeding+'\n')


# In[ ]:


# dict_user_id_embeeding[11]


# In[ ]:


save_embeeding(dict_user_id_embeeding, './embeeding/embeeding_user_id.txt')


# In[ ]:


# dict_user_id_embeeding.


# ## 抽取item_id的embeeding

# In[ ]:


dict_index_item_id_to_item_id = dict(zip(node_user_item.index_item_id,node_user_item.item_id))
dict_item_id_embeeding = {}
for index_item_id in set_index_item_id:
    item_id_embeeding = decompose.get_embeeding_item_id(feed_dict={decompose.input_item_id:[index_item_id]})
    dict_item_id_embeeding[dict_index_item_id_to_item_id[index_item_id]] = item_id_embeeding[0]


# In[ ]:


save_embeeding(dict_item_id_embeeding, './embeeding/embeeding_item_id.txt')


# In[ ]:


# !head ./embeeding/embeeding_item_id.txt


# In[ ]:


# !head ./embeeding/embeeding_user_id.txt


# In[ ]:


dict_item_id_embeeding[18]


# In[ ]:





# In[ ]:


node_user_item.head()


# In[ ]:


node_user_item.sample(10)


# In[ ]:


for i, row in node_user_item.sample(30).iterrows():
    item_id = int(row["item_id"])
    user_id = int(row["user_id"])
    score1 = np.sum(dict_user_id_embeeding[user_id] * dict_item_id_embeeding[item_id])
    
    sample_item_id = int(node_user_item.sample(1)["item_id"].values)
    score2 = np.sum(dict_user_id_embeeding[user_id] * dict_item_id_embeeding[sample_item_id])
    
    sample_user_id = int(node_user_item.sample(1)["user_id"].values)
    score3 = np.sum(dict_user_id_embeeding[sample_user_id] * dict_item_id_embeeding[item_id])
    
    print(score1, score2, score3, np.argmax([score1, score2, score3]))
    


# In[ ]:


# 保存 用户 和 商品特征


# In[ ]:





# ## LINE 分解

# In[ ]:


# !./tool/LINE-master/linux/line


# In[ ]:


# !ls "./line_model/"


# In[ ]:


# node_user_item[["item_id","user_id","score"]].to_csv("./line_model/net.txt")


# In[ ]:


# !./tool/LINE-master/linux/line -train ./line_model/net.txt -output ./line_model/vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025


# In[ ]:


# decompose.load("decompose.model")


# In[ ]:


# !ls ./line_model/


# ## 深度学习Baseline

# # 建模

# 构建训练数据

# In[ ]:


underexpose_train_click.head()


# In[ ]:


dict_train_user_phase_action[0][1]


# In[ ]:


underexpose_test_click.head()


# In[ ]:


dict_test_user_phase_action[0][11]


# In[ ]:


len(dict_test_user_phase_action[0][11]["item_seq"]),len(dict_test_user_phase_action[0][11]["diff_time_seq"])


# In[ ]:


[1,2,3][-1], [1,2,3][:-1]


# In[ ]:


([1,2,3] + [0])[:-1]


# In[ ]:


[1,2,3,4][:-3], [1,2,3,4][-3]


# In[ ]:


def get_train_test_data(
            dict_train_user_phase_action
            ,dict_test_user_phase_action
            ,list_sep_last_num
            ):
    list_seq_last_num, list_flag, list_phase, list_user_id, list_item_seq, list_diff_time, list_item_seq_next = [], [], [], [], [], [], []
    list_data = [(1, dict_train_user_phase_action), (0, dict_test_user_phase_action)]
    for sep_last_num in list_sep_last_num:
        for (flag, dict_user_phase_action) in list_data:
            for phase in dict_user_phase_action.keys():
                for user_id in dict_user_phase_action[phase].keys():
                    if sep_last_num < len(dict_user_phase_action[phase][user_id]["item_seq"]):
                        list_seq_last_num.append(sep_last_num)
                        list_flag.append(flag)
                        list_phase.append(phase)
                        list_user_id.append(user_id)
                        list_item_seq.append(dict_user_phase_action[phase][user_id]["item_seq"][:-sep_last_num])
                        list_diff_time.append((dict_user_phase_action[phase][user_id]["diff_time_seq"]+[0])[:-sep_last_num])
                        list_item_seq_next.append(dict_user_phase_action[phase][user_id]["item_seq"][-sep_last_num])
    underexpose_train_test_data = pd.DataFrame()
    underexpose_train_test_data["seq_last_num"] = list_seq_last_num
    underexpose_train_test_data["flag"] = list_flag
    underexpose_train_test_data["phase"] = list_phase
    underexpose_train_test_data["user_id"] = list_user_id
    underexpose_train_test_data["item_seq"] = list_item_seq
    underexpose_train_test_data["diff_time"] = list_diff_time
    underexpose_train_test_data["item_seq_next"] = list_item_seq_next
    underexpose_train_test_data["pred_if"] = underexpose_train_test_data["item_seq_next"].map(lambda x: 1 if x==-999 else 0)
    return underexpose_train_test_data 


# In[ ]:


underexpose_train_test_data = get_train_test_data(
                            dict_train_user_phase_action
                            ,dict_test_user_phase_action 
                            ,list_sep_last_num=[1,2,3]
                            )


# In[ ]:


underexpose_train_test_data.head()


# # 特征工程 + 辅助特征

# In[ ]:


underexpose_train_test_data_feat = underexpose_train_test_data.copy()


# In[ ]:


underexpose_train_test_data_feat.head()


# In[ ]:


underexpose_train_test_data_feat["item_seq_num"] = underexpose_train_test_data_feat["item_seq"].map(lambda x:len(x))
underexpose_train_test_data_feat["diff_time_max"] = underexpose_train_test_data_feat["diff_time"].map(lambda x:np.max(x))
underexpose_train_test_data_feat["diff_time_min"] = underexpose_train_test_data_feat["diff_time"].map(lambda x:np.min(x))
underexpose_train_test_data_feat["diff_time_mean"] = underexpose_train_test_data_feat["diff_time"].map(lambda x:np.mean(x))
underexpose_train_test_data_feat["diff_time_std"] = underexpose_train_test_data_feat["diff_time"].map(lambda x:np.std(x))


# In[ ]:


def diff2_time(diff_time):
    if len(diff_time) < 2:
        return [0]
    else:
        diff2_time = []
        for i in range(len(diff_time)-1):
            diff2_time.append(diff_time[i+1]-diff_time[i])
        return diff2_time


# In[ ]:


underexpose_train_test_data_feat["diff2_time"] = underexpose_train_test_data_feat["diff_time"].map(lambda x:diff2_time(x))


# In[ ]:


underexpose_train_test_data_feat["diff2_time_max"] = underexpose_train_test_data_feat["diff2_time"].map(lambda x:np.max(x))
underexpose_train_test_data_feat["diff2_time_min"] = underexpose_train_test_data_feat["diff2_time"].map(lambda x:np.min(x))
underexpose_train_test_data_feat["diff2_time_mean"] = underexpose_train_test_data_feat["diff2_time"].map(lambda x:np.mean(x))
underexpose_train_test_data_feat["diff2_time_std"] = underexpose_train_test_data_feat["diff2_time"].map(lambda x:np.std(x))


# In[ ]:


def get_item_seq_similarity(item_seq, dict_item_id_embeeding):
    item_seq_similarity = []
    for i in range(len(item_seq)-1):
        item_id_0 = item_seq[i]
        item_id_1 = item_seq[i+1]
        similarity = np.sum(dict_item_id_embeeding[item_id_0] * dict_item_id_embeeding[item_id_1]) / np.sqrt(np.sum(dict_item_id_embeeding[item_id_0]**2) * np.sum(dict_item_id_embeeding[item_id_1]**2))
        item_seq_similarity.append(similarity)
    return item_seq_similarity 


# In[ ]:


underexpose_train_test_data_feat["item_seq_similarity"] = underexpose_train_test_data_feat["item_seq"].map(lambda x:get_item_seq_similarity(x,dict_item_id_embeeding))


# In[ ]:





# In[ ]:


def get_user_item_seq_similarity(list_user_id, list_item_seq, dict_user_id_embeeding, dict_item_id_embeeding):
    list_user_id_item_seq_similarity = []
    for (user_id, item_seq) in zip(list_user_id, list_item_seq):
        user_id_item_seq_similarity = []
        for item_id in item_seq:
            similarity = np.sum(dict_user_id_embeeding[user_id] * dict_item_id_embeeding[item_id]) / np.sqrt(np.sum(dict_user_id_embeeding[user_id]**2) * np.sum(dict_item_id_embeeding[item_id]**2))
            user_id_item_seq_similarity.append(similarity)
        list_user_id_item_seq_similarity.append(user_id_item_seq_similarity)
    return list_user_id_item_seq_similarity


# In[ ]:


underexpose_train_test_data_feat["user_id_item_seq_similarity"] = get_user_item_seq_similarity(
                                                    underexpose_train_test_data_feat["user_id"]
                                                    ,underexpose_train_test_data_feat["item_seq"]
                                                    ,dict_user_id_embeeding
                                                    ,dict_item_id_embeeding)


# In[ ]:


underexpose_train_test_data_feat.head().head()


# In[ ]:


underexpose_train_test_data_feat.columns


# In[ ]:





# # 模型

# In[ ]:





# In[ ]:


def get_train_data(dict_user_action, sep_last_num):
    # user_id-sep_last_num, item_seq, diff_time_seq, item_seq_next
    # sep_last_num = -1, -2, -3
    list_index, list_user_id, list_item_seq, list_diff_time_seq, list_item_seq_next = [], [], [], [], []
    for user_id in dict_user_action.keys():
        if sep_last_num < len(dict_user_action[user_id]["item_seq"]):
            list_index.append(str(user_id) + "_" + str(sep_last_num))
            list_user_id.append(user_id)
            list_item_seq.append(dict_user_action[user_id]["item_seq"][:-sep_last_num])
            if 1-sep_last_num == 0:
                list_diff_time_seq.append(dict_user_action[user_id]["diff_time_seq"][:])
            else:
                list_diff_time_seq.append(dict_user_action[user_id]["diff_time_seq"][:1-sep_last_num])
            list_item_seq_next.append(dict_user_action[user_id]["item_seq"][-sep_last_num])
    df_train_data = pd.DataFrame()
    df_train_data["index"], df_train_data["user_id"], df_train_data["item_seq"], df_train_data["diff_time_seq"], df_train_data["item_seq_next"] = list_index, list_user_id, list_item_seq, list_diff_time_seq, list_item_seq_next
    return df_train_data


# In[ ]:





# In[ ]:


def get_pred_data(dict_user_action_test_click, test_qtime):
    # user_id-0, item_seq_diff_time_seq, pred_item_seq_next
    list_user_id, list_query_time, list_item_seq, list_diff_time_seq = [], [], [], []
    for i,row in test_qtime.iterrows(): 
        user_id, query_time = int(row["user_id"]), float(row["query_time"])
        list_user_id.append(user_id)
        list_query_time.append(query_time)
        list_item_seq.append(dict_user_action_test_click[user_id]["item_seq"])
        list_diff_time_seq.append(dict_user_action_test_click[user_id]["diff_time_seq"] + [(query_time-dict_user_action_test_click[user_id]["time_seq"][-1]) * 10**4])
    df_test_data = pd.DataFrame()
    df_test_data["user_id"], df_test_data["query_time"], df_test_data["item_seq"], df_test_data["diff_time_seq"] = list_user_id, list_query_time, list_item_seq, list_diff_time_seq
    return df_test_data


# In[ ]:





# In[ ]:


underexpose_test_click.sort_values(['user_id']).head()


# In[ ]:


underexpose_train_click.sort_values(['user_id']).head()


# In[ ]:


set(underexpose_train_click.user_id)-set(underexpose_user_feat.user_id)


# In[ ]:


node_user_item


# In[ ]:





# In[ ]:


# underexpose_user_feat.user_id.max()


# In[ ]:





# In[ ]:


pd.DataFrame


# In[ ]:


underexpose_train_click.head()


# In[ ]:


underexpose_test_click.head()


# In[ ]:





# # 建模

# 构建训练数据

# In[ ]:


def get_train_data(dict_user_action, sep_last_num):
    # user_id-sep_last_num, item_seq, diff_time_seq, item_seq_next
    # sep_last_num = -1, -2, -3
    list_index, list_user_id, list_item_seq, list_diff_time_seq, list_item_seq_next = [], [], [], [], []
    for user_id in dict_user_action.keys():
        if sep_last_num < len(dict_user_action[user_id]["item_seq"]):
            list_index.append(str(user_id) + "_" + str(sep_last_num))
            list_user_id.append(user_id)
            list_item_seq.append(dict_user_action[user_id]["item_seq"][:-sep_last_num])
            if 1-sep_last_num == 0:
                list_diff_time_seq.append(dict_user_action[user_id]["diff_time_seq"][:])
            else:
                list_diff_time_seq.append(dict_user_action[user_id]["diff_time_seq"][:1-sep_last_num])
            list_item_seq_next.append(dict_user_action[user_id]["item_seq"][-sep_last_num])
    df_train_data = pd.DataFrame()
    df_train_data["index"], df_train_data["user_id"], df_train_data["item_seq"], df_train_data["diff_time_seq"], df_train_data["item_seq_next"] = list_index, list_user_id, list_item_seq, list_diff_time_seq, list_item_seq_next
    return df_train_data


# In[ ]:


df_train_data = get_train_data(dict_user_action_train_click_0, 1)


# In[ ]:


df_train_data.head()


# In[ ]:


len(df_train_data.values[0][2])


# In[ ]:


len(df_train_data.values[0][3])


# In[ ]:


def get_pred_data(dict_user_action_test_click, test_qtime):
    # user_id-0, item_seq_diff_time_seq, pred_item_seq_next
    list_user_id, list_query_time, list_item_seq, list_diff_time_seq = [], [], [], []
    for i,row in test_qtime.iterrows(): 
        user_id, query_time = int(row["user_id"]), float(row["query_time"])
        list_user_id.append(user_id)
        list_query_time.append(query_time)
        list_item_seq.append(dict_user_action_test_click[user_id]["item_seq"])
        list_diff_time_seq.append(dict_user_action_test_click[user_id]["diff_time_seq"] + [(query_time-dict_user_action_test_click[user_id]["time_seq"][-1]) * 10**4])
    df_test_data = pd.DataFrame()
    df_test_data["user_id"], df_test_data["query_time"], df_test_data["item_seq"], df_test_data["diff_time_seq"] = list_user_id, list_query_time, list_item_seq, list_diff_time_seq
    return df_test_data


# In[ ]:


df_test_data = get_pred_data(dict_user_action_test_click_0, underexpose_test_qtime_0)


# ## 解题思路一

# In[ ]:





# ## 解题思路二

# In[ ]:





# ## 解题思路三

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




