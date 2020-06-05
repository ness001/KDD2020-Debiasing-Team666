import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from loguru import logger
import argparse


class PickleWrapper(object):

    @classmethod
    def loadFromFile(cls, file, mode='rb'):
        with open(file, mode) as f:
            return pickle.load(f)

    @classmethod
    def dump2File(cls, o, file, mode='wb'):
        '''
        把目标对象序列化到文件
        :param o: 目标对象
        :param file: 文件
        :param mode:
        :return:
        '''
        with open(file, mode) as f:
            pickle.dump(o, f)


parser = argparse.ArgumentParser(description='t')
parser.add_argument('--now_phase', type=int, default=6, help='')
parser.add_argument('--window', type=int, default=10, help='cocur_matr的时间窗')
parser.add_argument('--time_decay', type=float, default=7 / 8, help='时间衰减')
parser.add_argument('--mode', type=str, default='train', help='train test')
parser.add_argument('--topk', type=int, default=500, help='每种召回策略召回的样本数')
parser.add_argument('--DATA_DIR', type=str, default='../../data/', help='data dir')

args = parser.parse_args(args=[])
trace = logger.add(os.path.join(args.DATA_DIR, 'data_gen/runtime.log'))


# # Cell
def load_click_data_per_phase(now_phase, base_dir):
    """
    """
    train_path = os.path.join(base_dir, 'underexpose_train')
    test_path = os.path.join(base_dir, 'underexpose_test')

    all_click_df = []
    for c in range(now_phase + 1):
        logger.info(f'phase: {c}')
        cols_str = 'user_id item_id time'.split()
        click_train1 = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})
        click_test1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})
        test_qtime1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','time'], converters={c: str for c in cols_str})
        click_test1_val = click_test1.sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id'],keep='last')

        click_test1 = click_test1[~click_test1.index.isin(click_test1_val.index)]
        all_click = click_train1.append(click_test1).drop_duplicates().sort_values('time')

        all_click_df.append((all_click, click_test1_val, test_qtime1))
        logger.info(f'all_click: {all_click.shape}, click_test1_val: {click_test1_val.shape}, test_qtime1: {test_qtime1.shape}')
    return all_click_df
all_click_df = load_click_data_per_phase(args.now_phase, args.DATA_DIR)


def get_item_data():
    train_item_df = pd.read_csv(os.path.join(args.DATA_DIR, 'underexpose_train/underexpose_item_feat.csv'),
                                names=['item_id'] + ['vec' + str(i) for i in range(0, 256)],
                                converters={'item_id': str})
    train_item_df = train_item_df.replace('[\[\]]', '', regex=True)
    train_item_df.iloc[:, 1:] = train_item_df.iloc[:, 1:].astype(float)
    return train_item_df


item_feat = get_item_data()

item_feat['text_vec'] = item_feat.iloc[:, 1:129].values.tolist()
item_feat['img_vec'] = item_feat.iloc[:, 129:257].values.tolist()
item_feat['text_vec'] = item_feat['text_vec'].map(np.array)
item_feat['img_vec'] = item_feat['img_vec'].map(np.array)
item_feat.set_index('item_id', inplace=True)

def load_match_items():
    r_itemcf = PickleWrapper.loadFromFile(os.path.join('../../', 'user_data/r_list_itemcf_0527.pkl'))

    r_binn = PickleWrapper.loadFromFile(os.path.join('../../','user_data/r_list_binn_0527.pkl'))

#     r_itemcf_phase4 = PickleWrapper.loadFromFile(os.path.join(args.DATA_DIR, 'data_gen/r_list_itemcf_0527_phase4.pkl'))

#     r_binn_phase4 = PickleWrapper.loadFromFile(os.path.join(args.DATA_DIR, 'data_gen/r_list_binn_0527_phase4.pkl'))


    r_itemcf_yl = PickleWrapper.loadFromFile(os.path.join('../../', 'user_data/r_list_itemcf_yulao_0527.pkl'))

    # r_itemcf = r_itemcf + r_itemcf_phase4
    # r_binn = r_binn + r_binn_phase4
    r_itemcf_yl = [r_itemcf_yl[0], r_itemcf_yl[1],r_itemcf_yl[2],r_itemcf_yl[3],r_itemcf_yl[5],r_itemcf_yl[6],r_itemcf_yl[4]]
    # pahse 4?
    return r_itemcf, r_binn, r_itemcf_yl
#     return r_itemcf_phase4, r_binn_phase4, [r_itemcf_yl[4]]



match_num = 500
r_itemcf, r_binn, r_itemcf_yl = load_match_items()

def data_preporcess(recall_list, match_num, phase, mode='train'):
    itemcf, r_binn, r_itemcf_yl = recall_list
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    for i in range(len(phase)):
        r_itemcf[i]['phase'] = phase[i]
        r_binn[i]['phase'] = phase[i]
        r_itemcf_yl[i]['phase'] = phase[i]
        df1 = df1.append(r_itemcf[i])
        df2 = df2.append(r_binn[i])
        df3 = df3.append(r_itemcf_yl[i])

    df1_ = df1[df1['rank'] < (match_num + 1)]
    df2_ = df2[df2['rank'] < (match_num + 1)]
    df3_ = df3[df3['rank'] < (match_num + 1)]

    print('merge Multi-channel recall...')
    cols = 'user_id item_id_pred phase'.split()

    if mode == 'train':
        cols = 'user_id item_id_pred item_id_true phase'.split()
    df = pd.merge(pd.merge(df1_, df2_, on=cols, how='outer'),
                  df3_, on=cols, how='outer')

    df = df.fillna(0)
    if mode == 'train':
        df['label'] = (df['item_id_pred'] == df['item_id_true']).map(int)

    dft = pd.DataFrame()
    for p in tqdm(phase):
        # 获取test用户的近n次点击
        temp_ = all_click_df[p][0][all_click_df[p][0].user_id.isin(all_click_df[p][1].user_id)].groupby('user_id')['item_id'].agg(list)
        temp_ = pd.DataFrame(temp_)
        for i in range(0, 5):
            temp_[f'last_{i+1}'] = temp_.item_id.str.get(-(i+1)) #because time is ascending
        dft = dft.append(temp_.reset_index())
        # 计算每个phase的item_id cnt
        t_ = all_click_df[p][0].groupby('item_id')['user_id'].count() # item  hot degree in train set
        df.loc[df.phase==p, 'item_cnt'] = df.item_id_pred.map(lambda x: t_[x] if x in t_ else 0)

    df = pd.merge(df, dft.drop(columns='item_id'))

    df['item_id_pred_text_vec'] = df.item_id_pred.map(lambda x: item_feat['text_vec'][x] if x in item_feat['text_vec'] else np.zeros(128))

    df['item_id_pred_img_vec'] = df.item_id_pred.map(lambda x: item_feat['img_vec'][x] if x in item_feat['img_vec'] else np.arange(128))

    for c in tqdm('last_1 last_2 last_3'.split()):
        df[f'{c}_text_vec'] = df.item_id_pred.map(lambda x: item_feat['text_vec'][x] if x in item_feat['text_vec'] else np.zeros(128))

        df[f'{c}_img_vec'] = df.item_id_pred.map(lambda x: item_feat['img_vec'][x] if x in item_feat['img_vec'] else np.arange(128)) #why not np.zeros?


    # 太大了，不建议存储
    # df.to_pickle(os.path.join(args.DATA_DIR, f'data_gen/dft{match_num}.pkl'))

    # df = pd.read_pickle(os.path.join(args.DATA_DIR, f'data_gen/dft{match_num}.pkl'))
    return df

phase = [0, 1, 2, 3, 5, 6]
df = data_preporcess([r_itemcf, r_binn, r_itemcf_yl], match_num, phase)

def fe(df):
    for i in tqdm('rank score'.split()):
        df[f'{i}1_sub_{i}2'] = df[f'{i}_x']-df[f'{i}_y']
        df[f'{i}1_add_{i}2'] = df[f'{i}_x']+df[f'{i}_y']
        df[f'{i}1_mul_{i}2'] = df[f'{i}_x']*df[f'{i}_y']

        df[f'{i}1_sub_{i}3'] = df[f'{i}_x']-df[f'{i}']
        df[f'{i}1_add_{i}3'] = df[f'{i}_x']+df[f'{i}']
        df[f'{i}1_mul_{i}3'] = df[f'{i}_x']*df[f'{i}']

        df[f'{i}2_sub_{i}3'] = df[f'{i}_y']-df[f'{i}']
        df[f'{i}2_add_{i}3'] = df[f'{i}_y']+df[f'{i}']
        df[f'{i}2_mul_{i}3'] = df[f'{i}_y']*df[f'{i}']

    df['sim1_text'] = (df['item_id_pred_text_vec'] * df['last_1_text_vec']).map(sum)

    df['sim1_img'] = (df['item_id_pred_img_vec'] * df['last_1_img_vec']).map(sum)

    df['sim2_text'] = (df['item_id_pred_text_vec'] * df['last_2_text_vec']).map(sum)
    df['sim2_img'] = (df['item_id_pred_img_vec'] * df['last_2_img_vec']).map(sum)

    df['sim3_text'] = (df['item_id_pred_text_vec'] * df['last_3_text_vec']).map(sum)
    df['sim3_img'] = (df['item_id_pred_img_vec'] * df['last_3_img_vec']).map(sum)

#     df['sim4_text'] = (df['item_id_pred_text_vec'] * df['last_4_text_vec']).map(sum)
#     df['sim4_img'] = (df['item_id_pred_img_vec'] * df['last_4_img_vec']).map(sum)

#     df['sim5_text'] = (df['item_id_pred_text_vec'] * df['last_5_text_vec']).map(sum)
#     df['sim5_img'] = (df['item_id_pred_img_vec'] * df['last_5_img_vec']).map(sum)



    df['sim1_text_img'] = df['sim1_text'] *  df['sim1_img']
    df['sim2_text_img'] = df['sim2_text'] *  df['sim2_img']
    df['sim3_text_img'] = df['sim3_text'] *  df['sim3_img']
#     df['sim4_text_img'] = df['sim4_text'] *  df['sim4_img']
#     df['sim5_text_img'] = df['sim5_text'] *  df['sim5_img']

    df['sim12_text'] = df['sim1_text'] + df['sim2_text']
    df['sim123_text'] = df['sim1_text'] + df['sim2_text'] + df['sim3_text']
#     df['sim1234_text'] = df['sim1_text'] + df['sim2_text'] + df['sim3_text'] + df['sim4_text']
#     df['sim12345_text'] = df['sim1_text'] + df['sim2_text'] + df['sim3_text'] + df['sim4_text'] + df['sim5_text']

    df['sim12_img'] = df['sim1_img'] + df['sim2_img']
    df['sim123_img'] = df['sim1_img'] + df['sim2_img'] + df['sim3_img']
#     df['sim1234_img'] = df['sim1_img'] + df['sim2_img'] + df['sim3_img'] + df['sim4_img']
#     df['sim12345_img'] = df['sim1_img'] + df['sim2_img'] + df['sim3_img'] + df['sim4_img'] + df['sim5_img']

    df['sim12_text_img'] = df['sim1_text_img'] + df['sim2_text_img']
    df['sim123_text_img'] = df['sim1_text_img'] + df['sim2_text_img'] + df['sim3_text_img']
#     df['sim1234_text_img'] = df['sim1_text_img'] + df['sim2_text_img'] + df['sim3_text_img'] + df['sim4_text_img']
#     df['sim12345_text_img'] = df['sim1_text_img'] + df['sim2_text_img'] + df['sim3_text_img'] + df['sim4_text_img'] + df['sim5_text_img']


    return df

df = fe(df)