{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## install pacakges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple\n",
      "Collecting pandas==0.21\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9b/b7/d829de9794567443fbe230a666910d2c5ea3c28a6554f1246ab004583b82/pandas-0.21.0-cp36-cp36m-manylinux1_x86_64.whl (26.2 MB)\n",
      "\u001b[K     |################################| 26.2 MB 25.6 MB/s eta 0:00:01     |##########################      | 22.1 MB 25.6 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: numpy>=1.9.0 in /root/anaconda3/envs/python367/lib/python3.6/site-packages (from pandas==0.21) (1.18.4)\n",
      "Requirement already satisfied: pytz>=2011k in /root/anaconda3/envs/python367/lib/python3.6/site-packages (from pandas==0.21) (2020.1)\n",
      "Requirement already satisfied: python-dateutil>=2 in /root/anaconda3/envs/python367/lib/python3.6/site-packages (from pandas==0.21) (2.8.1)\n",
      "Requirement already satisfied: six>=1.5 in /root/anaconda3/envs/python367/lib/python3.6/site-packages (from python-dateutil>=2->pandas==0.21) (1.15.0)\n",
      "Installing collected packages: pandas\n",
      "  Attempting uninstall: pandas\n",
      "    Found existing installation: pandas 1.0.3\n",
      "    Uninstalling pandas-1.0.3:\n",
      "      Successfully uninstalled pandas-1.0.3\n",
      "Successfully installed pandas-0.21.0\n",
      "Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple\n",
      "Collecting tqdm\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f3/76/4697ce203a3d42b2ead61127b35e5fcc26bba9a35c03b32a2bd342a4c869/tqdm-4.46.1-py2.py3-none-any.whl (63 kB)\n",
      "\u001b[K     |################################| 63 kB 333 kB/s eta 0:00:011\n",
      "\u001b[?25hInstalling collected packages: tqdm\n",
      "Successfully installed tqdm-4.46.1\n",
      "Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple\n",
      "Collecting lightfm\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e9/8e/5485ac5a8616abe1c673d1e033e2f232b4319ab95424b42499fabff2257f/lightfm-1.15.tar.gz (302 kB)\n",
      "\u001b[K     |################################| 302 kB 885 kB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: numpy in /root/anaconda3/envs/python367/lib/python3.6/site-packages (from lightfm) (1.18.4)\n",
      "Requirement already satisfied: scipy>=0.17.0 in /root/anaconda3/envs/python367/lib/python3.6/site-packages (from lightfm) (1.4.1)\n",
      "Collecting requests\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1a/70/1935c770cb3be6e3a8b78ced23d7e0f3b187f5cbfab4749523ed65d7c9b1/requests-2.23.0-py2.py3-none-any.whl (58 kB)\n",
      "\u001b[K     |################################| 58 kB 1.3 MB/s  eta 0:00:01\n",
      "\u001b[?25hCollecting chardet<4,>=3.0.2\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bc/a9/01ffebfb562e4274b6487b4bb1ddec7ca55ec7510b22e4c51f14098443b8/chardet-3.0.4-py2.py3-none-any.whl (133 kB)\n",
      "\u001b[K     |################################| 133 kB 10.0 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e1/e5/df302e8017440f111c11cc41a6b432838672f5a70aa29227bf58149dc72f/urllib3-1.25.9-py2.py3-none-any.whl (126 kB)\n",
      "\u001b[K     |################################| 126 kB 10.3 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting idna<3,>=2.5\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/89/e3/afebe61c546d18fb1709a61bee788254b40e736cff7271c7de5de2dc4128/idna-2.9-py2.py3-none-any.whl (58 kB)\n",
      "\u001b[K     |################################| 58 kB 1.4 MB/s  eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: certifi>=2017.4.17 in /root/anaconda3/envs/python367/lib/python3.6/site-packages (from requests->lightfm) (2020.4.5.1)\n",
      "Building wheels for collected packages: lightfm\n",
      "  Building wheel for lightfm (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for lightfm: filename=lightfm-1.15-cp36-cp36m-linux_x86_64.whl size=748915 sha256=681c7a0c58b97bcba7a8c75f7628ba13236120ee3eef36410ae0d4c0bb7c1b26\n",
      "  Stored in directory: /root/.cache/pip/wheels/1f/69/66/d1b3c8124983aaa7fc6fe80e76052f0c303548b6bc45a66115\n",
      "Successfully built lightfm\n",
      "Installing collected packages: chardet, urllib3, idna, requests, lightfm\n",
      "Successfully installed chardet-3.0.4 idna-2.9 lightfm-1.15 requests-2.23.0 urllib3-1.25.9\n",
      "Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple\n",
      "Collecting loguru\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/80/b0/4413a201fcdcdc6789050c536d3b4ece601975ded9e0d676ef47f582348d/loguru-0.5.0-py3-none-any.whl (56 kB)\n",
      "\u001b[K     |################################| 56 kB 512 kB/s eta 0:00:011\n",
      "\u001b[?25hCollecting aiocontextvars>=0.2.0; python_version < \"3.7\"\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/db/c1/7a723e8d988de0a2e623927396e54b6831b68cb80dce468c945b849a9385/aiocontextvars-0.2.2-py2.py3-none-any.whl (4.9 kB)\n",
      "Collecting contextvars==2.4; python_version < \"3.7\"\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/83/96/55b82d9f13763be9d672622e1b8106c85acb83edd7cc2fa5bc67cd9877e9/contextvars-2.4.tar.gz (9.6 kB)\n",
      "Collecting immutables>=0.9\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/99/e0/ea6fd4697120327d26773b5a84853f897a68e33d3f9376b00a8ff96e4f63/immutables-0.14-cp36-cp36m-manylinux1_x86_64.whl (98 kB)\n",
      "\u001b[K     |################################| 98 kB 1.1 MB/s eta 0:00:011\n",
      "\u001b[?25hBuilding wheels for collected packages: contextvars\n",
      "  Building wheel for contextvars (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for contextvars: filename=contextvars-2.4-py3-none-any.whl size=7664 sha256=b45e072e2952d0101b8f6c093a7bfe7c38f99e52143190950e428ba46103a929\n",
      "  Stored in directory: /root/.cache/pip/wheels/b2/89/2f/8d9cd55f6bdd54305f96b76eb998218d275992a694c3a887a0\n",
      "Successfully built contextvars\n",
      "Installing collected packages: immutables, contextvars, aiocontextvars, loguru\n",
      "Successfully installed aiocontextvars-0.2.2 contextvars-2.4 immutables-0.14 loguru-0.5.0\n"
     ]
    }
   ],
   "source": [
    "!pip install pandas==0.21\n",
    "!pip install tqdm \n",
    "!pip install lightfm\n",
    "!pip install loguru"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## load data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2020-06-07 10:44:45.866 | INFO     | __main__:load_click_data_per_phase:36 - phase: 0\n",
      "2020-06-07 10:44:47.166 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (261337, 3), click_test1_val: (1663, 3), test_qtime1: (1663, 2)\n",
      "2020-06-07 10:44:47.168 | INFO     | __main__:load_click_data_per_phase:36 - phase: 1\n",
      "2020-06-07 10:44:48.488 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (264871, 3), click_test1_val: (1726, 3), test_qtime1: (1726, 2)\n",
      "2020-06-07 10:44:48.491 | INFO     | __main__:load_click_data_per_phase:36 - phase: 2\n",
      "2020-06-07 10:44:49.781 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (264624, 3), click_test1_val: (1690, 3), test_qtime1: (1690, 2)\n",
      "2020-06-07 10:44:49.783 | INFO     | __main__:load_click_data_per_phase:36 - phase: 3\n",
      "2020-06-07 10:44:51.175 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (286609, 3), click_test1_val: (1675, 3), test_qtime1: (1675, 2)\n",
      "2020-06-07 10:44:51.177 | INFO     | __main__:load_click_data_per_phase:36 - phase: 4\n",
      "2020-06-07 10:44:52.586 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (291672, 3), click_test1_val: (1708, 3), test_qtime1: (1708, 2)\n",
      "2020-06-07 10:44:52.589 | INFO     | __main__:load_click_data_per_phase:36 - phase: 5\n",
      "2020-06-07 10:44:54.130 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (313378, 3), click_test1_val: (1798, 3), test_qtime1: (1798, 2)\n",
      "2020-06-07 10:44:54.132 | INFO     | __main__:load_click_data_per_phase:36 - phase: 6\n",
      "2020-06-07 10:44:55.796 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (337339, 3), click_test1_val: (1821, 3), test_qtime1: (1821, 2)\n",
      "2020-06-07 10:44:55.798 | INFO     | __main__:load_click_data_per_phase:36 - phase: 7\n",
      "2020-06-07 10:44:57.250 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (299152, 3), click_test1_val: (1797, 3), test_qtime1: (1797, 2)\n",
      "2020-06-07 10:44:57.252 | INFO     | __main__:load_click_data_per_phase:36 - phase: 8\n",
      "2020-06-07 10:44:58.654 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (292828, 3), click_test1_val: (1818, 3), test_qtime1: (1818, 2)\n",
      "2020-06-07 10:44:58.657 | INFO     | __main__:load_click_data_per_phase:36 - phase: 9\n",
      "2020-06-07 10:45:00.018 | INFO     | __main__:load_click_data_per_phase:47 - all_click: (281588, 3), click_test1_val: (1752, 3), test_qtime1: (1752, 2)\n"
     ]
    }
   ],
   "source": [
    "from IPython.core.interactiveshell import InteractiveShell\n",
    "InteractiveShell.ast_node_interactivity = 'all'\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from tqdm import tqdm\n",
    "import pickle\n",
    "from loguru import logger\n",
    "import argparse\n",
    "import multiprocessing\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "parser = argparse.ArgumentParser(description='t')\n",
    "parser.add_argument('--now_phase', type=int, default=9, help='')\n",
    "parser.add_argument('--window', type=int, default=10, help='cocur_matr的时间窗')\n",
    "parser.add_argument('--time_decay', type=float, default=7/8, help='时间衰减')\n",
    "parser.add_argument('--mode', type=str, default='train', help='train test')\n",
    "parser.add_argument('--topk', type=int, default=500, help='每种召回策略召回的样本数')\n",
    "parser.add_argument('--DATA_DIR', type=str, default='./', help='data dir')\n",
    "\n",
    "args = parser.parse_args(args=[])\n",
    "trace = logger.add(os.path.join(args.DATA_DIR, 'data_gen/runtime.log'))\n",
    "\n",
    "# Cell\n",
    "def load_click_data_per_phase(now_phase, base_dir):\n",
    "    \"\"\"\n",
    "    \"\"\"\n",
    "\n",
    "\n",
    "    all_click_df = []\n",
    "    for c in range(now_phase + 1):\n",
    "        logger.info(f'phase: {c}')\n",
    "        cols_str = 'user_id item_id time'.split()\n",
    "        click_train1 = pd.read_csv( './underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})\n",
    "        click_test1 = pd.read_csv( './underexpose_test_click-{}.csv'.format(c, c), header=None,  names=['user_id', 'item_id', 'time'], converters={c: str for c in cols_str})\n",
    "        test_qtime1 = pd.read_csv( './underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','time'], converters={c: str for c in cols_str})\n",
    "        click_test1_val = click_test1.sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id'],keep='last')\n",
    "\n",
    "        click_test1 = click_test1[~click_test1.index.isin(click_test1_val.index)]\n",
    "        all_click = click_train1.append(click_test1).drop_duplicates().sort_values('time')\n",
    "\n",
    "        all_click_df.append((all_click, click_test1_val, test_qtime1))\n",
    "        logger.info(f'all_click: {all_click.shape}, click_test1_val: {click_test1_val.shape}, test_qtime1: {test_qtime1.shape}')\n",
    "    return all_click_df\n",
    "all_click_df = load_click_data_per_phase(args.now_phase, args.DATA_DIR)\n",
    "all_train=pd.concat([all_click_df[i][0] for i in range(0,7)]).drop_duplicates().reset_index(drop=True)\n",
    "all_train=all_train.sort_values(by=['user_id','time'],ascending=['True','False'])\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "col_name=['item_id']\n",
    "for i in range(0,128):\n",
    "    col_name.append('tv'+str(i))\n",
    "for i in range(0,128):\n",
    "    col_name.append('iv'+str(i))\n",
    "itemft=pd.read_csv('./underexpose_item_feat.csv',low_memory=False,names=col_name)\n",
    "itemft=itemft.replace('[\\[\\]]','',regex=True)#regex=True is the key\n",
    "itemft=itemft.astype({'item_id':int, 'tv0':float, 'tv127':float, 'iv0':float, 'iv127':float})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "userft=pd.read_csv('./underexpose_user_feat.csv',low_memory=False,names=['user_id','age','gender','city'],dtype=\\\n",
    "                  {'user_id': int,'age':'object','gender':'object','city':'object'})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# user feat have duplicated user ids\n",
    "a=userft.user_id.value_counts()\n",
    "dup_user_id=a[a>1].index.tolist()\n",
    "dup_index=userft.loc[userft.user_id.isin(dup_user_id),:].index.tolist()\n",
    "dup_index=[i for i in dup_index if i%2 == 0]\n",
    "userft.drop(dup_index,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## get a phase"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "p=8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_train=all_click_df[p][0].astype(float)\n",
    "all_test=all_click_df[p][1].astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(292828, 3)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_last1=all_click_df[p-1][0].astype(float)\n",
    "train_last2=all_click_df[p-2][0].astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(299152, 3)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_last1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(403811, 3)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_train=pd.concat([all_train,train_last1]).drop_duplicates()\n",
    "all_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## prepare train test coo_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(292828, 3)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1818, 3)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(1817, 3)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_test.shape\n",
    "all_test=all_test[(all_test['user_id'].isin(all_train['user_id'])) & (all_test['item_id'].isin(all_train['item_id']))]\n",
    "all_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "le_cols=['item_id','user_id']\n",
    "encoded_train=dict()\n",
    "encoded_test=dict()\n",
    "for key in le_cols:\n",
    "    le=LabelEncoder()\n",
    "    encoded_train[key]=le.fit_transform(all_train[key].values)\n",
    "    encoded_test[key]=le.transform(all_test[key].values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'item_id': array([ 4045, 37623,  2269, ..., 39835, 39330, 37617]),\n",
       " 'user_id': array([ 6631, 19125,  6417, ...,  1602, 17094,  3304])}"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "encoded_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(19883, 44978)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_users,n_items=len(np.unique(encoded_train['user_id'])),len(np.unique(encoded_train['item_id']))\n",
    "\n",
    "n_users,n_items\n",
    "\n",
    "train_codata=np.ones(shape=(all_train.shape[0],))\n",
    "test_codata=np.ones(shape=(all_test.shape[0],))\n",
    "\n",
    "from scipy.sparse import coo_matrix\n",
    "\n",
    "train = coo_matrix((train_codata,(encoded_train['user_id'],encoded_train['item_id'])),shape=(n_users,n_items))\n",
    "test = coo_matrix((test_codata,(encoded_test['user_id'],encoded_test['item_id'])),shape=(n_users,n_items))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## prepare userft itemft"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 19883 entries, 0 to 19882\n",
      "Data columns (total 4 columns):\n",
      "user_id    19883 non-null float64\n",
      "age        19883 non-null object\n",
      "gender     19883 non-null object\n",
      "city       19883 non-null object\n",
      "dtypes: float64(1), object(3)\n",
      "memory usage: 776.7+ KB\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 44978 entries, 0 to 44977\n",
      "Columns: 257 entries, item_id to iv127\n",
      "dtypes: float64(257)\n",
      "memory usage: 88.5 MB\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(19883, 3)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(19883, 3)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(44978, 256)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(44978, 256)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(19883, 3)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(44978, 256)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "user_item_matrix=all_train.pivot_table(index='user_id',columns='item_id',aggfunc=len,fill_value=0)\n",
    "users=pd.DataFrame(data=list(user_item_matrix.index),columns=['user_id'])\n",
    "user_features=users.merge(userft,how='left')\n",
    "user_features.shape[0] == user_item_matrix.shape[0]\n",
    "\n",
    "\n",
    "items=pd.DataFrame(data=list(user_item_matrix.columns.droplevel(level=0)),columns=['item_id'])\n",
    "item_features=items.merge(itemft,how='left')\n",
    "item_features.shape[0] == user_item_matrix.shape[1]\n",
    "\n",
    "\n",
    "user_features.fillna(0,inplace=True)\n",
    "item_features.fillna(0,inplace=True)\n",
    "\n",
    "user_features.info()\n",
    "item_features.info()\n",
    "\n",
    "user_features=user_features.replace({'M':1,'F':0})\n",
    "\n",
    "user_features=user_features.astype({'user_id':int,'age':int,'gender':int,'city':int})\n",
    "\n",
    "user_features.set_index('user_id',inplace=True)\n",
    "item_features.set_index('item_id',inplace=True)\n",
    "\n",
    "user_features.dropna().shape\n",
    "user_features.shape\n",
    "item_features.dropna().shape\n",
    "item_features.shape\n",
    "\n",
    "\n",
    "from scipy.sparse import csr_matrix\n",
    "uft_csr=csr_matrix(user_features.values)\n",
    "ift_csr=csr_matrix(item_features.values)\n",
    "uft_csr.shape\n",
    "ift_csr.shape\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "del user_item_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## abort: user feature is too sparse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Not all estimated parameters are finite, your model may have diverged. Try decreasing the learning rate or normalising feature values and sample weights",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-51-66c7e33da6ef>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mlightfm\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mLightFM\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mLightFM\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlearning_rate\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'bpr'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0muser_features\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0muft_csr\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mitem_features\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mift_csr\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mepochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mverbose\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/anaconda3/envs/python367/lib/python3.6/site-packages/lightfm/lightfm.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, interactions, user_features, item_features, sample_weight, epochs, num_threads, verbose)\u001b[0m\n\u001b[1;32m    477\u001b[0m                                 \u001b[0mepochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mepochs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    478\u001b[0m                                 \u001b[0mnum_threads\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnum_threads\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 479\u001b[0;31m                                 verbose=verbose)\n\u001b[0m\u001b[1;32m    480\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    481\u001b[0m     def fit_partial(self, interactions,\n",
      "\u001b[0;32m~/anaconda3/envs/python367/lib/python3.6/site-packages/lightfm/lightfm.py\u001b[0m in \u001b[0;36mfit_partial\u001b[0;34m(self, interactions, user_features, item_features, sample_weight, epochs, num_threads, verbose)\u001b[0m\n\u001b[1;32m    576\u001b[0m                             self.loss)\n\u001b[1;32m    577\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 578\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_finite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    579\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    580\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/python367/lib/python3.6/site-packages/lightfm/lightfm.py\u001b[0m in \u001b[0;36m_check_finite\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    411\u001b[0m             \u001b[0;31m# large boolean temporary.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    412\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0misfinite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mparameter\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 413\u001b[0;31m                 raise ValueError(\"Not all estimated parameters are finite,\"\n\u001b[0m\u001b[1;32m    414\u001b[0m                                  \u001b[0;34m\" your model may have diverged. Try decreasing\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    415\u001b[0m                                  \u001b[0;34m\" the learning rate or normalising feature values\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: Not all estimated parameters are finite, your model may have diverged. Try decreasing the learning rate or normalising feature values and sample weights"
     ]
    }
   ],
   "source": [
    "from lightfm.data import Dataset\n",
    "# Dataset.build_item_features(data=uft_csr)\n",
    "from lightfm import LightFM\n",
    "model=LightFM(learning_rate=5*10**(-10),loss='bpr')\n",
    "model.fit(train,user_features=uft_csr,item_features=ift_csr,epochs=3,verbose=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "error: ValueError: Not all estimated parameters are finite, your model may have diverged. Try decreasing the learning rate or normalising feature values and sample weights\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.isnan(np.min(user_features.values))\n",
    "np.isnan(np.min(item_features.values))\n",
    "np.isnan(np.min(train))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## set useful tool"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fm_eval(coo,prec_k=10,recall_k=100,use='itemft',train_interactions=None,ift_csr=None,t=1):\n",
    "    from lightfm.evaluation import precision_at_k,auc_score,recall_at_k\n",
    "    if use == 'itemft':\n",
    "        train_precision = precision_at_k(model,coo,train_interactions=train_interactions,item_features=ift_csr,k=prec_k,num_threads=t).mean()\n",
    "        train_auc = auc_score(model,coo,train_interactions=train_interactions,item_features=ift_csr,num_threads=t).mean()\n",
    "        train_recall=recall_at_k(model,coo,train_interactions=train_interactions,k=recall_k,item_features=ift_csr,num_threads=t).mean()\n",
    "        print('train: prec: {} with top {},auc: {} , recall: {} with top {}'.format(train_precision,prec_k,train_auc,train_recall,recall_k))\n",
    "    if use == None:\n",
    "        train_precision = precision_at_k(model,coo,train_interactions=train_interactions,k=prec_k,num_threads=t).mean()\n",
    "        train_auc = auc_score(model,coo,train_interactions=train_interactions,num_threads=t).mean()\n",
    "        train_recall=recall_at_k(model,coo,train_interactions=train_interactions,k=recall_k,num_threads=t).mean()\n",
    "        print('train: prec: {} with top {},auc: {} , recall: {} with top {}'.format(train_precision,prec_k,train_auc,train_recall,recall_k))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "import multiprocessing\n",
    "num_threads=multiprocessing.cpu_count()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## with no feature"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### with le constructed train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f55a03d0240>"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0006053934921510518 with top 10,auc: 0.7194451093673706 , recall: 0.08145294441386902 with top 500\n"
     ]
    }
   ],
   "source": [
    "model=LightFM(learning_schedule='adadelta',loss='warp',random_state=400)\n",
    "model.fit(interactions=train,epochs=50,verbose=False,num_threads=num_threads-1,item_features=None)\n",
    "fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f55a03d0320>"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.00033021465060301125 with top 10,auc: 0.6331506371498108 , recall: 0.07099614749587231 with top 500\n"
     ]
    }
   ],
   "source": [
    "model=LightFM(learning_schedule='adagrad',loss='warp',random_state=400)\n",
    "model.fit(interactions=train,epochs=50,verbose=False,num_threads=num_threads-1,item_features=None)\n",
    "fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/1 [00:00<?, ?it/s]\n",
      "  0%|          | 0/3 [00:00<?, ?it/s]\u001b[A"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "lr: 0.028\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f55a0a5a630>"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      " 33%|███▎      | 1/3 [00:29<00:58, 29.11s/it]\u001b[A"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0003852504014503211 with top 10,auc: 0.5993421077728271 , recall: 0.06329113924050633 with top 500\n",
      "lr: 0.028\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f55a0a01828>"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      " 67%|██████▋   | 2/3 [00:58<00:29, 29.05s/it]\u001b[A"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0005503577413037419 with top 10,auc: 0.6142310500144958 , recall: 0.06714364336818933 with top 500\n",
      "lr: 0.028\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f55a111b2b0>"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "100%|██████████| 3/3 [01:27<00:00, 29.06s/it]\u001b[A\n",
      "100%|██████████| 1/1 [01:27<00:00, 87.19s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0005503577413037419 with top 10,auc: 0.5946887135505676 , recall: 0.06053935057787562 with top 500\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "#0.28,0.067 with only current phase data \n",
    "# with 400 rs recall:0.07\n",
    "from lightfm import LightFM\n",
    "for lr in tqdm([0.028]):\n",
    "    for rs in tqdm([400,500,600]):\n",
    "        print('lr: {}'.format(lr))\n",
    "        model=LightFM(learning_rate=lr,loss='warp',random_state=rs)\n",
    "        model.fit(interactions=train,epochs=50,verbose=False,num_threads=num_threads-1,item_features=None)\n",
    "        fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use=None)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7ff6125ca240>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0007234279764816165 with top 10,auc: 0.6581267714500427 , recall: 0.08124652198107958 with top 500\n"
     ]
    }
   ],
   "source": [
    "\n",
    "model=LightFM(learning_schedule='adagrad',loss='warp')\n",
    "model.fit(interactions=train,epochs=100,verbose=False,num_threads=num_threads-1,item_features=None)\n",
    "fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.10661077499389648 with top 1,auc: 0.924675703048706 , recall: 0.010644894253196598 with top 1\n"
     ]
    }
   ],
   "source": [
    "fm_eval(train,recall_k=1,prec_k=1,use=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.1086977869272232 with top 1,auc: 0.9098623394966125 , recall: 0.2905786382196576 with top 500\n"
     ]
    }
   ],
   "source": [
    "fm_eval(train,recall_k=500,prec_k=1,use=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0007234279764816165 with top 10,auc: 0.6346980333328247 , recall: 0.08514190317195326 with top 500\n"
     ]
    }
   ],
   "source": [
    "fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0012242626398801804 with top 10,auc: 0.6131116151809692 , recall: 0.0027824151363383415 with top 1\n"
     ]
    }
   ],
   "source": [
    "model.item_biases *= 0.0\n",
    "fm_eval(test,train_interactions=train,recall_k=1,prec_k=10,use=None)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### only on train with simple pivot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.sparse import coo_matrix\n",
    "coo=coo_matrix(user_item_matrix)\n",
    "coo.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "from lightfm import LightFM\n",
    "model=LightFM(learning_rate=0.05,loss='bpr')\n",
    "model.fit(interactions=coo,epochs=50,verbose=False,num_threads=num_threads-1,item_features=None)\n",
    "fm_eval(coo,recall_k=1,prec_k=1,use=None)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## with only item feature ***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### with le constructed train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "47"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from lightfm import LightFM\n",
    "import multiprocessing\n",
    "num_threads=multiprocessing.cpu_count()-1\n",
    "num_threads"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fm_eval(coo,prec_k=10,recall_k=100,use='itemft',train_interactions=None,ift_csr=None,t=1):\n",
    "    from lightfm.evaluation import precision_at_k,auc_score,recall_at_k\n",
    "    if use == 'itemft':\n",
    "        train_precision = precision_at_k(model,coo,train_interactions=train_interactions,item_features=ift_csr,k=prec_k,num_threads=t).mean()\n",
    "        train_auc = auc_score(model,coo,train_interactions=train_interactions,item_features=ift_csr,num_threads=t).mean()\n",
    "        train_recall=recall_at_k(model,coo,train_interactions=train_interactions,k=recall_k,item_features=ift_csr,num_threads=t).mean()\n",
    "        print('train: prec: {} with top {},auc: {} , recall: {} with top {}'.format(train_precision,prec_k,train_auc,train_recall,recall_k))\n",
    "    if use == None:\n",
    "        train_precision = precision_at_k(model,coo,train_interactions=train_interactions,k=prec_k,num_threads=t).mean()\n",
    "        train_auc = auc_score(model,coo,train_interactions=train_interactions,num_threads=t).mean()\n",
    "        train_recall=recall_at_k(model,coo,train_interactions=train_interactions,k=recall_k,num_threads=t).mean()\n",
    "        print('train: prec: {} with top {},auc: {} , recall: {} with top {}'.format(train_precision,prec_k,train_auc,train_recall,recall_k))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0\n",
      "Epoch 1\n",
      "Epoch 2\n",
      "Epoch 3\n",
      "Epoch 4\n",
      "Epoch 5\n",
      "Epoch 6\n",
      "Epoch 7\n",
      "Epoch 8\n",
      "Epoch 9\n",
      "Epoch 10\n",
      "Epoch 11\n",
      "Epoch 12\n",
      "Epoch 13\n",
      "Epoch 14\n",
      "Epoch 15\n",
      "Epoch 16\n",
      "Epoch 17\n",
      "Epoch 18\n",
      "Epoch 19\n",
      "Epoch 20\n",
      "Epoch 21\n",
      "Epoch 22\n",
      "Epoch 23\n",
      "Epoch 24\n",
      "Epoch 25\n",
      "Epoch 26\n",
      "Epoch 27\n",
      "Epoch 28\n",
      "Epoch 29\n",
      "Epoch 30\n",
      "Epoch 31\n",
      "Epoch 32\n",
      "Epoch 33\n",
      "Epoch 34\n",
      "Epoch 35\n",
      "Epoch 36\n",
      "Epoch 38\n",
      "Epoch 39\n",
      "Epoch 40\n",
      "Epoch 41\n",
      "Epoch 42\n",
      "Epoch 43\n",
      "Epoch 44\n",
      "Epoch 45\n",
      "Epoch 46\n",
      "Epoch 47\n",
      "Epoch 48\n",
      "Epoch 49\n",
      "train: prec: 0.0 with top 1,auc: 0.14192229509353638 , recall: 0.01375894331315355 with top 500\n",
      "CPU times: user 6h 53min 55s, sys: 15.1 s, total: 6h 54min 10s\n",
      "Wall time: 37min 52s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "model=LightFM(no_components=128,learning_schedule='adagrad',loss='warp')\n",
    "model.fit(train,epochs=50,item_features=ift_csr,verbose=True,num_threads=num_threads-1)\n",
    "fm_eval(test,train_interactions=train,recall_k=500,prec_k=1,use='itemft',ift_csr=ift_csr,t=num_threads-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0 with top 1,auc: 0.14152708649635315 , recall: 0.015960374243258118 with top 500\n"
     ]
    }
   ],
   "source": [
    "model.item_biases *= 0.0\n",
    "fm_eval(test,train_interactions=train,recall_k=500,prec_k=1,use='itemft',ift_csr=ift_csr,t=num_threads-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0\n",
      "Epoch 1\n",
      "Epoch 2\n",
      "Epoch 3\n",
      "Epoch 4\n",
      "Epoch 5\n",
      "Epoch 6\n",
      "Epoch 7\n",
      "Epoch 8\n",
      "Epoch 9\n",
      "Epoch 10\n",
      "Epoch 11\n",
      "Epoch 12\n",
      "Epoch 13\n",
      "Epoch 14\n",
      "Epoch 15\n",
      "Epoch 16\n",
      "Epoch 17\n",
      "Epoch 18\n",
      "Epoch 19\n",
      "Epoch 20\n",
      "Epoch 21\n",
      "Epoch 22\n",
      "Epoch 23\n",
      "Epoch 24\n",
      "Epoch 25\n",
      "Epoch 26\n",
      "Epoch 27\n",
      "Epoch 28\n",
      "Epoch 29\n",
      "train: prec: 0.0 with top 1,auc: 0.1366032510995865 , recall: 0.013208585580627407 with top 500\n",
      "CPU times: user 5h 40min 59s, sys: 14.1 s, total: 5h 41min 13s\n",
      "Wall time: 31min 11s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "model=LightFM(no_components=128,learning_rate=0.2,loss='warp')\n",
    "model.fit(train,epochs=30,item_features=ift_csr,verbose=True,num_threads=num_threads-1)\n",
    "fm_eval(test,train_interactions=train,recall_k=500,prec_k=1,use='itemft',ift_csr=ift_csr,t=num_threads-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "with open('./hybrid{}.pkl'.format(p),'wb') as f:\n",
    "    pickle.dump(model,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/2 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0\n",
      "Epoch 1\n",
      "Epoch 2\n",
      "Epoch 3\n",
      "Epoch 4\n",
      "Epoch 5\n",
      "Epoch 6\n",
      "Epoch 7\n",
      "Epoch 8\n",
      "Epoch 9\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7ff6150c2fd0>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 50%|█████     | 1/2 [24:03<24:03, 1443.33s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.0005008347216062248 with top 10,auc: 0.7351633310317993 , recall: 0.10907067334446299 with top 500\n",
      "Epoch 0\n",
      "Epoch 1\n",
      "Epoch 2\n",
      "Epoch 3\n",
      "Epoch 4\n",
      "Epoch 5\n",
      "Epoch 6\n",
      "Epoch 7\n",
      "Epoch 8\n",
      "Epoch 9\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7ff6125ca160>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2/2 [47:22<00:00, 1421.13s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train: prec: 0.000612131436355412 with top 10,auc: 0.7319753170013428 , recall: 0.10740122426266 with top 500\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# for lr in tqdm([0.005,0.015]):\n",
    "#     model=LightFM(no_components=32,learning_rate=0.01,loss='warp')\n",
    "#     model.fit(train,epochs=10,item_features=ift_csr,verbose=True,num_threads=num_threads-1)\n",
    "#     fm_eval(test,train_interactions=train,recall_k=500,prec_k=10,use='itemft',ift_csr=ift_csr,t=num_threads-1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## all_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_mapping_dict(le):\n",
    "    d={}\n",
    "    for c in le.classes_:\n",
    "        d.update({c:le.transform([c])[0]})\n",
    "    return d\n",
    "\n",
    "\n",
    "le_cols=['item_id','user_id']\n",
    "encoded_train=dict()\n",
    "encoded_test=dict()\n",
    "for key in le_cols:\n",
    "    le=LabelEncoder()\n",
    "    encoded_train[key]=le.fit_transform(all_train[key].values)\n",
    "    encoded_test[key]=le.transform(all_test[key].values)\n",
    "    if key=='item_id':\n",
    "        item_dict=get_mapping_dict(le)\n",
    "    else:\n",
    "        user_dict=get_mapping_dict(le)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(1.0, 0),\n",
       " (2.0, 1),\n",
       " (3.0, 2),\n",
       " (4.0, 3),\n",
       " (5.0, 4),\n",
       " (7.0, 5),\n",
       " (8.0, 6),\n",
       " (9.0, 7),\n",
       " (10.0, 8),\n",
       " (13.0, 9)]"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "[(3.0, 0),\n",
       " (9.0, 1),\n",
       " (14.0, 2),\n",
       " (16.0, 3),\n",
       " (18.0, 4),\n",
       " (23.0, 5),\n",
       " (28.0, 6),\n",
       " (29.0, 7),\n",
       " (31.0, 8),\n",
       " (32.0, 9)]"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from itertools import islice\n",
    "def take(n, iterable):\n",
    "    \"Return first n items of the iterable as a list\"\n",
    "    return list(islice(iterable, n))\n",
    "take(10, user_dict.items())\n",
    "take(10, item_dict.items())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1821/1821 [13:39<00:00,  2.22it/s]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "recall_frame=pd.DataFrame()\n",
    "topn=500\n",
    "item_history=all_train.groupby('user_id')['item_id'].agg(lambda x: list(x))\n",
    "recall_frame=pd.DataFrame()\n",
    "topn=500\n",
    "\n",
    "\n",
    "for user in tqdm(all_test.user_id.unique()):\n",
    "    user_code=user_dict[user]\n",
    "    scores=pd.Series(model.predict(user_code,np.arange(n_items),item_features=ift_csr))\n",
    "    scores.index=user_item_matrix.columns.droplevel(0)\n",
    "    all_recall = pd.Series(scores.sort_values(ascending=False))\n",
    "    history = item_history[user]\n",
    "    recall_items = [x for x in all_recall.index if x not in history][0:topn]\n",
    "    recall_scores=all_recall.loc[recall_items].values\n",
    "    n=len(recall_items)\n",
    "\n",
    "    true_item=list(all_test.loc[all_test.user_id==user].item_id.values)\n",
    "\n",
    "    df=pd.DataFrame({'rank':range(1,n+1),'user': [user]*n, 'scores':recall_scores,'item_pred':recall_items,'item_true':true_item*n} )\n",
    "    recall_frame=pd.concat([recall_frame,df])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>item_pred</th>\n",
       "      <th>item_true</th>\n",
       "      <th>rank</th>\n",
       "      <th>scores</th>\n",
       "      <th>user</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>81508.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>1</td>\n",
       "      <td>8.721670</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>21923.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>2</td>\n",
       "      <td>8.465993</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>94797.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>3</td>\n",
       "      <td>8.377849</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>885.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>4</td>\n",
       "      <td>8.302359</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>87621.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>5</td>\n",
       "      <td>8.141705</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6879.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>6</td>\n",
       "      <td>8.133941</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>97241.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>7</td>\n",
       "      <td>8.037683</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>105419.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>8</td>\n",
       "      <td>7.942738</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>78507.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>9</td>\n",
       "      <td>7.892912</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>88028.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>10</td>\n",
       "      <td>7.889555</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>89180.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>11</td>\n",
       "      <td>7.883130</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>94780.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>12</td>\n",
       "      <td>7.777550</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>68282.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>13</td>\n",
       "      <td>7.747831</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>34836.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>14</td>\n",
       "      <td>7.679563</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>6321.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>15</td>\n",
       "      <td>7.613523</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>19447.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>16</td>\n",
       "      <td>7.533205</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>98091.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>17</td>\n",
       "      <td>7.530992</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>72890.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>18</td>\n",
       "      <td>7.491286</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>53854.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>19</td>\n",
       "      <td>7.379835</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>94803.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>20</td>\n",
       "      <td>7.365088</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>41340.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>21</td>\n",
       "      <td>7.245782</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>13247.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>22</td>\n",
       "      <td>7.138545</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>63520.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>23</td>\n",
       "      <td>7.134003</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>43979.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>24</td>\n",
       "      <td>7.133385</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>14681.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>25</td>\n",
       "      <td>7.116866</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>83985.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>26</td>\n",
       "      <td>7.113834</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>4607.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>27</td>\n",
       "      <td>7.112077</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>32973.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>28</td>\n",
       "      <td>7.040993</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>7124.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>29</td>\n",
       "      <td>7.013610</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>228.0</td>\n",
       "      <td>43173.0</td>\n",
       "      <td>30</td>\n",
       "      <td>6.994743</td>\n",
       "      <td>10005.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>470</th>\n",
       "      <td>19986.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>471</td>\n",
       "      <td>0.479825</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>471</th>\n",
       "      <td>79153.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>472</td>\n",
       "      <td>0.476699</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>472</th>\n",
       "      <td>18335.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>473</td>\n",
       "      <td>0.473176</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>473</th>\n",
       "      <td>106536.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>474</td>\n",
       "      <td>0.471878</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>474</th>\n",
       "      <td>99763.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>475</td>\n",
       "      <td>0.469872</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>475</th>\n",
       "      <td>91733.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>476</td>\n",
       "      <td>0.468940</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>476</th>\n",
       "      <td>2186.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>477</td>\n",
       "      <td>0.467077</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>477</th>\n",
       "      <td>48074.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>478</td>\n",
       "      <td>0.463863</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>478</th>\n",
       "      <td>3606.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>479</td>\n",
       "      <td>0.462261</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>479</th>\n",
       "      <td>24328.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>480</td>\n",
       "      <td>0.461847</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>480</th>\n",
       "      <td>96709.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>481</td>\n",
       "      <td>0.459538</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>481</th>\n",
       "      <td>1447.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>482</td>\n",
       "      <td>0.459060</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>482</th>\n",
       "      <td>54148.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>483</td>\n",
       "      <td>0.455563</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>483</th>\n",
       "      <td>83545.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>484</td>\n",
       "      <td>0.454133</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>484</th>\n",
       "      <td>46036.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>485</td>\n",
       "      <td>0.451103</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>485</th>\n",
       "      <td>34709.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>486</td>\n",
       "      <td>0.450551</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>486</th>\n",
       "      <td>6695.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>487</td>\n",
       "      <td>0.450088</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>487</th>\n",
       "      <td>1249.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>488</td>\n",
       "      <td>0.449754</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>488</th>\n",
       "      <td>4601.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>489</td>\n",
       "      <td>0.449084</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>489</th>\n",
       "      <td>109885.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>490</td>\n",
       "      <td>0.447828</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>490</th>\n",
       "      <td>21010.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>491</td>\n",
       "      <td>0.447648</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>491</th>\n",
       "      <td>85822.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>492</td>\n",
       "      <td>0.447118</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>492</th>\n",
       "      <td>706.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>493</td>\n",
       "      <td>0.446685</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>493</th>\n",
       "      <td>21978.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>494</td>\n",
       "      <td>0.444715</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>494</th>\n",
       "      <td>46110.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>495</td>\n",
       "      <td>0.442793</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>495</th>\n",
       "      <td>5631.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>496</td>\n",
       "      <td>0.438138</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>496</th>\n",
       "      <td>30018.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>497</td>\n",
       "      <td>0.436086</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>497</th>\n",
       "      <td>51391.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>498</td>\n",
       "      <td>0.435992</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>498</th>\n",
       "      <td>52178.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>499</td>\n",
       "      <td>0.433109</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>499</th>\n",
       "      <td>19244.0</td>\n",
       "      <td>921.0</td>\n",
       "      <td>500</td>\n",
       "      <td>0.433079</td>\n",
       "      <td>9994.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>910500 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     item_pred  item_true  rank    scores     user\n",
       "0      81508.0    43173.0     1  8.721670  10005.0\n",
       "1      21923.0    43173.0     2  8.465993  10005.0\n",
       "2      94797.0    43173.0     3  8.377849  10005.0\n",
       "3        885.0    43173.0     4  8.302359  10005.0\n",
       "4      87621.0    43173.0     5  8.141705  10005.0\n",
       "5       6879.0    43173.0     6  8.133941  10005.0\n",
       "6      97241.0    43173.0     7  8.037683  10005.0\n",
       "7     105419.0    43173.0     8  7.942738  10005.0\n",
       "8      78507.0    43173.0     9  7.892912  10005.0\n",
       "9      88028.0    43173.0    10  7.889555  10005.0\n",
       "10     89180.0    43173.0    11  7.883130  10005.0\n",
       "11     94780.0    43173.0    12  7.777550  10005.0\n",
       "12     68282.0    43173.0    13  7.747831  10005.0\n",
       "13     34836.0    43173.0    14  7.679563  10005.0\n",
       "14      6321.0    43173.0    15  7.613523  10005.0\n",
       "15     19447.0    43173.0    16  7.533205  10005.0\n",
       "16     98091.0    43173.0    17  7.530992  10005.0\n",
       "17     72890.0    43173.0    18  7.491286  10005.0\n",
       "18     53854.0    43173.0    19  7.379835  10005.0\n",
       "19     94803.0    43173.0    20  7.365088  10005.0\n",
       "20     41340.0    43173.0    21  7.245782  10005.0\n",
       "21     13247.0    43173.0    22  7.138545  10005.0\n",
       "22     63520.0    43173.0    23  7.134003  10005.0\n",
       "23     43979.0    43173.0    24  7.133385  10005.0\n",
       "24     14681.0    43173.0    25  7.116866  10005.0\n",
       "25     83985.0    43173.0    26  7.113834  10005.0\n",
       "26      4607.0    43173.0    27  7.112077  10005.0\n",
       "27     32973.0    43173.0    28  7.040993  10005.0\n",
       "28      7124.0    43173.0    29  7.013610  10005.0\n",
       "29       228.0    43173.0    30  6.994743  10005.0\n",
       "..         ...        ...   ...       ...      ...\n",
       "470    19986.0      921.0   471  0.479825   9994.0\n",
       "471    79153.0      921.0   472  0.476699   9994.0\n",
       "472    18335.0      921.0   473  0.473176   9994.0\n",
       "473   106536.0      921.0   474  0.471878   9994.0\n",
       "474    99763.0      921.0   475  0.469872   9994.0\n",
       "475    91733.0      921.0   476  0.468940   9994.0\n",
       "476     2186.0      921.0   477  0.467077   9994.0\n",
       "477    48074.0      921.0   478  0.463863   9994.0\n",
       "478     3606.0      921.0   479  0.462261   9994.0\n",
       "479    24328.0      921.0   480  0.461847   9994.0\n",
       "480    96709.0      921.0   481  0.459538   9994.0\n",
       "481     1447.0      921.0   482  0.459060   9994.0\n",
       "482    54148.0      921.0   483  0.455563   9994.0\n",
       "483    83545.0      921.0   484  0.454133   9994.0\n",
       "484    46036.0      921.0   485  0.451103   9994.0\n",
       "485    34709.0      921.0   486  0.450551   9994.0\n",
       "486     6695.0      921.0   487  0.450088   9994.0\n",
       "487     1249.0      921.0   488  0.449754   9994.0\n",
       "488     4601.0      921.0   489  0.449084   9994.0\n",
       "489   109885.0      921.0   490  0.447828   9994.0\n",
       "490    21010.0      921.0   491  0.447648   9994.0\n",
       "491    85822.0      921.0   492  0.447118   9994.0\n",
       "492      706.0      921.0   493  0.446685   9994.0\n",
       "493    21978.0      921.0   494  0.444715   9994.0\n",
       "494    46110.0      921.0   495  0.442793   9994.0\n",
       "495     5631.0      921.0   496  0.438138   9994.0\n",
       "496    30018.0      921.0   497  0.436086   9994.0\n",
       "497    51391.0      921.0   498  0.435992   9994.0\n",
       "498    52178.0      921.0   499  0.433109   9994.0\n",
       "499    19244.0      921.0   500  0.433079   9994.0\n",
       "\n",
       "[910500 rows x 5 columns]"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.17792421746293247"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(recall_frame.item_pred == recall_frame.item_true)/1821"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [],
   "source": [
    "recall_frame.to_csv('./fm_recall-{}.csv'.format(p))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### only on train with simple pivot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.sparse import coo_matrix\n",
    "coo=coo_matrix(user_item_matrix)\n",
    "coo.shape\n",
    "coo.shape[0] == user_features.shape[0]\n",
    "coo.shape[1] == item_features.shape[0]\n",
    "from lightfm import LightFM\n",
    "model=LightFM(learning_rate=0.05,loss='bpr')\n",
    "model.fit(coo,epochs=10,item_features=ift_csr,verbose=True,num_threads=num_threads-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## plot learning schedule"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160da0>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<lightfm.lightfm.LightFM at 0x7f651d160128>"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "ename": "NameError",
     "evalue": "name 'plt' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-145-92c05ea972d5>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     27\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0madagrad_auc\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 29\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0madagrad_auc\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     30\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0madadelta_auc\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     31\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlegend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'adagrad'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'adadelta'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mloc\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'lower right'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'plt' is not defined"
     ]
    }
   ],
   "source": [
    "alpha = 1e-3\n",
    "epochs = 70\n",
    "\n",
    "adagrad_model = LightFM(no_components=30,\n",
    "                        loss='warp',\n",
    "                        learning_schedule='adagrad',\n",
    "                        user_alpha=alpha,\n",
    "                        item_alpha=alpha)\n",
    "adadelta_model = LightFM(no_components=30,\n",
    "                        loss='warp',\n",
    "                        learning_schedule='adadelta',\n",
    "                        user_alpha=alpha,\n",
    "                        item_alpha=alpha)\n",
    "\n",
    "adagrad_auc = []\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    adagrad_model.fit_partial(train, epochs=1)\n",
    "    adagrad_auc.append(auc_score(adagrad_model, test).mean())\n",
    "\n",
    "\n",
    "adadelta_auc = []\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    adadelta_model.fit_partial(train, epochs=1)\n",
    "    adadelta_auc.append(auc_score(adadelta_model, test).mean())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7f651dc16b70>]"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7f65c91c3e80>]"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x7f65c8dcf6d8>"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "x = np.arange(len(adagrad_auc))\n",
    "plt.plot(x, np.array(adagrad_auc))\n",
    "plt.plot(x, np.array(adadelta_auc))\n",
    "plt.legend(['adagrad', 'adadelta'], loc='lower right')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "with open('./user_item_matrix.pkl','wb') as f:\n",
    "    pickle.dump(user_item_matrix,f,protocol=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "recall_frame=pd.read_csv('./fm_recall.csv').iloc[:,1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.04612850082372323"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "0.07303679297089512"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "0.17792421746293247"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_50=recall_frame.loc[recall_frame['rank'] <=50]\n",
    "recall_100=recall_frame.loc[recall_frame['rank']<=100]\n",
    "recall_500=recall_frame.loc[recall_frame['rank']<=500]\n",
    "\n",
    "\n",
    "sum(recall_50.item_pred == recall_50.item_true)/1821\n",
    "sum(recall_100.item_pred == recall_100.item_true )/1821\n",
    "sum(recall_500.item_pred == recall_500.item_true )/1821\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}