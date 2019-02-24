import pandas as pd
import numpy as np
import joblib
import os
import datetime as dt
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

# 填写路径
TEST_PATH = './data/FuSai.csv'
TRAIN_PATH = './data/FuSai_train.csv'
TEST_NAME = 'FuSai'

#*********************** define functions ************************#
def fillna_A2_A3(df):
    '''
    描述：
        填充A2,A3的缺失值为0
    '''
    data = df.copy()
    tmp = data[data['A2'].isnull()]
    data.loc[data['A2'].isnull(), 'A2'] = 0
    
    tmp = data[data['A3'].isnull()]
    data.loc[data['A3'].isnull(),'A3'] = 0
    return data

def calc_hours(time_str):
    if time_str != time_str: # pd.isnull()的判断思路是一样的
            return time_str
    if type(time_str) != str:
        time_str = str(time_str)
    if time_str.split(':')[-1] == '':
        time_str = time_str + '00'
    if time_str.split(':')[0] == '':
        time_str = '00' + time_str 
    if len(time_str.split(':'))==2:
        time_str = time_str + ':00'    
    if time_str == '24:00:00':
        time_str = '23:59:00'
    if time_str == '0:00:00':
        time_str = '23:59:00'
    if time_str == '0:0:00':
        time_str = '23:59:00'
    try:
        time_obj = dt.datetime.strptime(time_str, '%H:%M:%S')
        hours = time_obj.hour + time_obj.minute/60 + time_obj.second/3600
    except:
        hours = 0
    return hours


def calc_s_interval(data, apply_cols=[]):
    '''
    描述：
        计算时间间隔
    '''
    df = data.copy()
    
    def process_time_str(time_str, pos):
        if time_str != time_str: # pd.isnull()的判断思路是一样的
            return time_str
        new_time_str = str(time_str).split('-')[pos]
        return calc_hours(new_time_str)
    
    for col in apply_cols:
        # 拆分始末时刻,并转换成 hour单位
        tmp = df[col]
        start_arr = tmp.apply(process_time_str, pos=0)
        end_arr = tmp.apply(process_time_str, pos=1)
        # 计算时间间隔
        interval_arr = end_arr - start_arr
        if np.sum(interval_arr<0)>0:
            print(col,'存在跨越一天的情况！')
        interval_arr[interval_arr<0] = interval_arr[interval_arr<0] + 24 # 跨越一天
        df[col+'_start'] = start_arr
        df[col+'_interval'] = interval_arr
        df[col+'_end'] = end_arr
    df.drop(columns=apply_cols, inplace=True)
    return df

def calc_time(data, apply_cols=[]):
    '''
    描述：
        把时间戳转化为 hour单位
    '''
    df = data.copy()
    for col in apply_cols:
        tmp = df[col]
        df[col+'_start'] = tmp.apply(lambda x: calc_hours(x))
        
    df.drop(columns=apply_cols, inplace=True)
    return df

def trans_relative_time(data, time_cols=[]):
    '''
    描述：
        把时间戳转化为距离开始时刻的相对时间
    '''
    df = data.copy()
    last_time = df[time_cols[0]] # 时间戳
    last_total_delta = 0
    for col in time_cols:
        delta = df[col].values - last_time.values 
        null_cond = np.isnan(delta)
        delta[null_cond] = 0 
        
        if sum(delta<0) > 0:
#             print(col,'存在跨越一天的情况!')
            delta[delta<0] = delta[delta<0] + 24 # 跨越一天
        
        # 更新last_time
        tmp = last_time.values[null_cond]
        last_time = df[col].copy() # 时间戳
        last_time.loc[null_cond] = tmp
        
        df[col] = delta + last_total_delta # 时间戳转化为相对时间
        
        # 更新last_total_delta
        last_total_delta = last_total_delta + delta
        
        # 还原缺失部分
        df.loc[null_cond, col] = None
    return df

#******************************特征工程*****************************#
# 1.加载、合并train、test_A
try:
    train = pd.read_csv(open(TRAIN_PATH, mode='r', encoding='gbk'))
except:
    train = pd.read_csv(TRAIN_PATH)
    
# A25处理
try:
    A25_cond = train['A25'] != '1900/3/10 0:00'
    train = train[A25_cond]
    train.reset_index(drop=True, inplace=True)
except:
    pass
train['A25'] = train['A25'].astype('float64')

testA = pd.read_csv(open(TEST_PATH, mode='r', encoding='gbk'))
LEN_TEST = testA.shape[0]
# train = train.drop(columns=['收率'])
testA['收率'] = 0


# 2.去除异常值(train)
train = train[(train['收率']==0) | ((train['收率']>0.86)&(train['收率']<1))] # 0.86~1.00之间
train = train.reset_index(drop=True)
data_ad = pd.concat([train,testA], axis=0)
data_ad.reset_index(drop=True, inplace=True)
data_ad['样本id'] = data_ad['样本id'].apply(lambda x: int(str(x).split('_')[1]))  
data_ad['样本id'] = data_ad['样本id'].astype(int)

# 3.去除异常列(both)
data_ad = data_ad.drop(columns=['A7','A8']) # t1, T1

# 4.fillna A2,A3(both)
data_A2A3 = fillna_A2_A3(data_ad)

# 5. time interval()
data_intvl = calc_s_interval(data_A2A3, apply_cols=['A20', 'A28', 'B4', 'B9', 'B10','B11'])

# 6.转换时间格式
data_time = calc_time(data_intvl, ['A5','A9','A11','A14','A16','A24','A26',
                     'B5','B7'])



# 7.处理s14、s15的缺失值
# 查看s13,s14,s15的数据分布
# 规律：如果累计甩滤用时>3h，之后的过程就不再进行

data_time.loc[:,['B9_start', 'B9_end',  
             'B10_start', 'B10_end',  
             'B11_start', 'B11_end' ]] = data_time[['B9_start', 'B9_end',  
                                                         'B10_start', 'B10_end',  
                                                         'B11_start', 'B11_end' ]].fillna(method='ffill', axis=1)
data_time.drop(columns=['B9_interval', 'B9_end', 'B10_start', 'B10_interval', 'B10_end', 'B11_start', 'B11_interval'], inplace=True)

# 8.调整列名顺序， drop A2('氢氧化钠')
data_time = data_time[['A1', 'A2', 'A3', 'A4',
                       'A5_start', 'A6', 
#                        'A7_start', 'A8', 
                       'A9_start', 'A10',
                       'A11_start', 'A12', 'A13', 
                       'A14_start', 'A15', 
                       'A16_start', 'A17', 'A18', 
                       'A19',
                       'A20_start','A20_interval', 'A20_end', 
                       'A21', 'A22', 'A23', 'A24_start', 'A25',  'A26_start', 'A27', 
                       'A28_start', 'A28_interval', 'A28_end',
                       'B1', 'B2', 'B3', 'B4_start', 'B4_interval', 'B4_end', 'B5_start', 'B6', 
                       'B7_start', 'B8', 
                       'B9_start', 'B11_end', 
                       'B12', 'B13', 
                       'B14',   
                        '收率','样本id']] 


# 输出参数的范围
if not os.path.exists('./pipeline_data'):
    os.mkdir('./pipeline_data')
if not os.path.exists('./pipeline_data/prior_data.csv'):
    
    data_pipeline = data_time.copy()
    data_pipeline['B9_B10_B11_interval'] = data_pipeline['B11_end'] - data_pipeline['B9_start']
    data_pipeline.loc[data_pipeline['B9_B10_B11_interval']<0, 'B9_B10_B11_interval'] = \
            data_pipeline.loc[data_pipeline['B9_B10_B11_interval']<0, 'B9_B10_B11_interval'] + 24
    
    keys =['A1', 'A2', 'A3', 'A4',
                           'A5_start', 'A6', 
    #                        'A7_start', 'A8', 
                           'A9_start', 'A10',
                           'A11_start', 'A12', 'A13', 
                           'A14_start', 'A15', 
                           'A16_start', 'A17', 'A18', 
                           'A19',
                           'A20_start','A20_interval', 'A21', 'A22', 'A23', 'A24_start', 'A25', 
                           'A26_start', 'A27', 
                           'A28_start', 'A28_interval', 
                           'B1', 'B2', 'B3', 
                           'B4_start', 'B4_interval', 'B5_start', 'B6', 
                           'B7_start', 'B8', 
    #                        'B9_start', 'B9_interval', 'B9_end', 
    #                        'B10_start', 'B10_interval', 'B10_end',
    #                         'B11_start', 'B11_interval', 'B11_end', 
                            'B9_start', 'B9_B10_B11_interval',
                           'B12', 'B13', 
                           'B14']
    data_pipeline = data_pipeline.loc[:train.shape[0]-1,keys]
    data_pipeline.to_csv('./pipeline_data/prior_data.csv', index=False)

# 修改列名为有意义的字段
data_time.columns = ['4-氰基砒啶', '氢氧化钠', '氢氧化钠溶液', '纯化水(原料)', # 第1次投料
                     't0', 'T0',
#                      't1', 'T1',
                     't2', 'T2',
                     't3', 'T3', 'P3',
                     't4', 'T4',
                     't5', 'T5', 'P5',
                     '纯化水(投料)', 's6_s', 's6_i', 's6_e',  # 第2次投料
                     '脱色原料1', '脱色原料2', '脱色原料3', 't7', 'T7', 't8', 'T8', # 第3次投料, 脱色过程， 起始和终止时刻
                     's9_s', 's9_i', 's9_e', # 去除脱色物质过程
                     '神秘物质(盐酸)', '神秘物质(盐酸)浓度', '神秘物质(盐酸)滴加后的酸碱浓度', 's10_s', 's10_i', 's10_e', 't11', 'T11', # 第4次投料滴加过程
                     't12', 'T12',
                     's13_s', 's15_e',  
                     '滴加物质', '滴加物质的浓度', # 第5次投料,作用于S13-S15,400/90min
                     '神秘物质(纯化水)',
                     'target','样本id']



# 9.转换时刻为相对时间
rev_apply_cols = ['t0', 't2', 't3','t4','t5',
                     't7', 
                     't8',
                     's9_s', 's9_e',
                     't11',
                     't12', 
                     's13_s', 's15_e']
data_revtime = trans_relative_time(data_time, rev_apply_cols)



# 剩余少量缺失处理
# 23条样本直接ffill填充
# null_cond = data_revtime['神秘物质(盐酸)'].isnull()|\
#               data_revtime['脱色原料1'].isnull()|\
#               data_revtime['脱色原料3'].isnull()|\
#               data_revtime['神秘物质(盐酸)浓度'].isnull()|\
#               data_revtime['神秘物质(盐酸)滴加后的酸碱浓度'].isnull()|\
#               data_revtime['t7'].isnull()|\
#               data_revtime['T7'].isnull()|\
#               data_revtime['t8'].isnull()|\
#               data_revtime['T8'].isnull()|\
#               data_revtime['t11'].isnull()|\
#               data_revtime['T12'].isnull()|\
#               data_revtime['滴加物质'].isnull()|\
#               data_revtime['滴加物质的浓度'].isnull()

# data_revtime.fillna(method='ffill', inplace=True)

# 10.精简过程
df = data_revtime.copy()
all_drop_cond = np.array([False]*df.shape[0])
result_df = df.copy()

# 过程1：搅拌加热过程
# *原料：(4-氰基砒啶,氢氧化钠溶液,纯化水(原料))=>  m1
# 氢氧化钠溶液_4-氰基砒啶, 氢氧化钠_4-氰基砒啶, 纯化水(原料)_4-氰基砒啶
# *af_1

result_df['氢氧化钠溶液_4-氰基砒啶'] = df['氢氧化钠溶液'].values / df['4-氰基砒啶'].values
result_df['氢氧化钠_4-氰基砒啶'] = df['氢氧化钠'].values / df['4-氰基砒啶'].values
result_df['纯化水(原料)_4-氰基砒啶'] = df['纯化水(原料)'].values / df['4-氰基砒啶'].values
result_df['af_1'] = (df['T2'] - df['T0']).values / ((df['t2']-df['t0']).values * df[['4-氰基砒啶','氢氧化钠', '氢氧化钠溶液','纯化水(原料)']].sum(axis=1).values)


# 过程2：水解过程
# t2_t5_intvl, T2_T5_mean
T2_T5_mean, t2_t5_intvl= df[['T2','T3','T4','T5']].mean(axis=1).values, df['t5'].values - df['t2'].values

result_df['T2_T5_mean'] = T2_T5_mean
result_df['t2_t5_intvl'] = t2_t5_intvl


# 过程3: 补水过程
# 原料: 纯化水(投料) => m2


# 过程4：脱色过程
# *原料：(脱色原料1，脱色原料2，脱色原料3) => m3
# T7_T8_mean, t7_t8_intvl

T7_T8_mean, t7_t8_intvl = df[['T7','T8']].mean(axis=1).values, df['t8'].values - df['t7'].values
result_df['T7_T8_mean'] = T7_T8_mean
result_df['t7_t8_intvl'] = t7_t8_intvl


# 过程5：去除脱色物质过程
# s9_i
result_df['s9_i'] = df['s9_i']
result_df['s9_s_t8_intvl'] = df['s9_s'] - df['t8']
#*程序A*#
# 4-氰基砒啶 + 氢氧化钠溶液 + 纯化水(原料) +  纯化水(投料) + 脱色原料1 + 脱色原料2 + 脱色原料3 =>mA
# T0_T8_delta, t0_t8_intvl, T0_T8_delta/t0_t8_intvl
result_df['mA'] = df[['4-氰基砒啶','氢氧化钠溶液', '纯化水(原料)' , '纯化水(投料)' ,'脱色原料1' ,'脱色原料2', '脱色原料3']].sum(axis=1)
result_df['T0_T8_delta'] = df['T8'].values - df['T0'].values
result_df['t0_t8_intvl'] = df['t8'].values - df['t0'].values
result_df['T0_T8_delta/t0_t8_intvl'] = result_df['T0_T8_delta'].values/result_df['t0_t8_intvl'].values

# 过程6：神秘物质（盐酸）滴加过程 
# 神秘物质(盐酸)*神秘物质(盐酸)浓度=> m4
# 神秘物质(盐酸)滴加后的酸碱浓度*mA/m4 => h4  酸度变化率
# s10_i, T11_T12_delta, t11_t12_intvl
m4 = df['神秘物质(盐酸)'].values * df['神秘物质(盐酸)浓度'].values
h4 = (df['神秘物质(盐酸)滴加后的酸碱浓度'].values * result_df['mA'].values) / m4

T11_T12_delta, t11_t12_intvl = df['T12'].values - df['T11'].values, df['t12'].values - df['t11'].values

result_df['m4'] = m4
result_df['h4'] = h4
result_df['T11_T12_delta'] = T11_T12_delta
result_df['t11_t12_intvl'] = t11_t12_intvl

result_df['s10_i'] = df['s10_i'] 

# 过程7：甩滤过程
# 原料：滴加物质*滴加物质的浓度 => m5
# s13_s_s15_e_intvl
result_df['m5'] = df['滴加物质'].values * df['滴加物质的浓度'].values
result_df['s13_s_s15_e_intvl'] = df['s15_e'].values - df['s13_s'].values 
# result_df.loc[result_df['s13_s_s15_e_intvl']<0, 's13_s_s15_e_intvl'] = result_df.loc[result_df['s13_s_s15_e_intvl']<0, 's13_s_s15_e_intvl'] + 24

# 过程8：神秘物质
# 原料：纯化水 
# 'af2'
result_df['af2'] = result_df['神秘物质(纯化水)'].values / result_df['mA'].values
#* 总体统计 *#
result_df['t0_t12_intvl'] = df['t12'].values - df['t0'].values
# result_df.loc[result_df['t0_t12_intvl']<0, 't0_t12_intvl'] = result_df.loc[result_df['t0_t12_intvl']<0, 't0_t12_intvl'] + 24

result_df['t0_s15_e_intvl'] = df['s15_e'].values - df['t0'].values
# result_df.loc[result_df['t0_s15_e_intvl']<0, 't0_s15_e_intvl'] = result_df.loc[result_df['t0_s15_e_intvl']<0, 't0_s15_e_intvl'] + 24

# 在完成时间的计算之后，修改t0
result_df['t0'] = data_time['t0']
data_refine = result_df

# 11.删除无用的列
data_refine.drop(columns=[ 's6_e'], inplace=True)
data_refine.drop(columns=[ 's9_e'], inplace=True)
data_refine.drop(columns=[ 's10_e'], inplace=True)
data_refine.drop(columns=[ 's15_e'], inplace=True)

# *12.删除重复率高的特征
# rate_list = []
# for col in data_refine.columns:
#     rate = data_refine[col].value_counts(normalize=True, dropna=False).values[0]
#     rate_list.append(rate)
# vc = pd.Series(data=rate_list, index=data_refine.columns)
# left_cols = vc[vc<0.9].index
# data_left = data_refine[left_cols].copy()
data_left = data_refine.copy()


# 基本数据处理完毕
train = data_left[:train.shape[0]].copy()
test  = data_left[train.shape[0]:].copy()
test.drop(columns=['target'], inplace=True)


new_train = train.copy()
new_train = new_train.sort_values(['样本id'], ascending=True)
train_copy = train.copy()
train_copy = train_copy.sort_values(['样本id'], ascending=True)

# 把train加长两倍, 构成闭环
train_len = len(new_train)
new_train = pd.concat([new_train, train_copy])

# 把加长两倍的train拼接到test后面
new_test = test.copy()
new_test = pd.concat([new_test, new_train.drop(columns=['target'])], axis=0, sort=False)

#***************开始向后做差***********************#
train_sort_ids = train_copy['样本id'].values

# 构造新的训练集
def gen_new_train(i):
    # print('构造作差的训练集')
    diff_tmp = new_train.diff(periods=-i, axis=0)
    diff_tmp = diff_tmp[:train_len]
    diff_tmp.columns = [col_ + '_difference' for col_ in
                        diff_tmp.columns.values]
    # 求完差值后加上样本id
    diff_tmp['样本id'] = train_sort_ids
    return diff_tmp

if not os.path.exists('./train_data'):
    os.mkdir('./train_data')
    
if not os.path.exists('./train_data/diff_train.lz4'):
    diff_tmp_list = Parallel(n_jobs=-1, verbose=1)(delayed(gen_new_train)(i) for i in range(1, train_len))
    diff_train = pd.concat(diff_tmp_list, axis=0, sort=False)

# 构造新的测试集
test_ids = test['样本id'].values
test_len = len(test)

def gen_new_test(i):
    # print('构造作差的测试集')
    diff_tmp = new_test.diff(periods=-i, axis=0)
    diff_tmp = diff_tmp[:test_len]
    diff_tmp.columns = [col_ + '_difference' for col_ in
                        diff_tmp.columns.values]
    diff_tmp['样本id'] = test_ids
    return diff_tmp

diff_tmp_list = Parallel(n_jobs=10, verbose=10)(delayed(gen_new_test)(i)\
    for i in range(test_len, test_len+train_len))                    

diff_test = pd.concat(diff_tmp_list, axis=0, sort=False)


# 和train顺序一致的target
train_tmp = train.copy()

train_target = train_tmp['target']
train_tmp.drop(['target'], axis=1, inplace=True)

# 拼接原始特征
if not os.path.exists('./train_data/diff_train.lz4'):
    diff_train = pd.merge(diff_train, train_tmp, how='left', on='样本id')
    diff_target = diff_train['target_difference']
    diff_train.drop(['target_difference'], axis=1, inplace=True)
    joblib.dump(diff_train, './train_data/diff_train.lz4', compress='lz4')
    joblib.dump(diff_target, './train_data/diff_target.lz4', compress='lz4')
else:
    diff_train = joblib.load('./train_data/diff_train.lz4')
    diff_target = joblib.load('./train_data/diff_target.lz4')

diff_test = pd.merge(diff_test, test, how='left', on='样本id')


# 使用 target_difference 作为训练目标
import time
X_train = diff_train.drop(columns=['样本id']).copy()
y_train = diff_target
X_test = diff_test.drop(columns=['样本id']).copy()


param = {'num_leaves': 31, #31
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         # "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l2": 0.1,
         # "lambda_l1": 0.1,
         'num_thread':12,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(diff_train))
predictions_lgb = np.zeros(len(diff_test))


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/lgb_model_%d'%fold_):
        print('train model')
        dev = X_train.iloc[trn_idx]
        val = X_train.iloc[val_idx]
        trn_data = lgb.Dataset(dev, y_train.iloc[trn_idx])
        val_data = lgb.Dataset(val, y_train.iloc[val_idx])

        num_round = 3000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=5,
                    early_stopping_rounds=100)
        joblib.dump(clf, './model/lgb_model_%d'%fold_)
    else:
        a = time.time()
        print('load model')
        val = X_train.iloc[val_idx]
        clf = joblib.load('./model/lgb_model_%d'%fold_)
        print('load time:%.2f s'%(time.time() - a))
    oof_lgb[val_idx] = clf.predict(val, num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits


# 还原train target
diff_train['compare_id'] = diff_train['样本id'].values - diff_train['样本id_difference'].values

# 提取训练集'样本id','target',用于作为参照数据
base_df = pd.DataFrame()
base_df['compare_id'] = train['样本id'].values
base_df['compare_target'] = list(train_target)


if not os.path.exists('./train_data'):
    os.mkdir('./train_data')

if not os.path.exists('./train_data/result_train_FuSai.csv'):
    # 把做差的target拼接回去
    diff_train = pd.merge(diff_train, base_df, how='left', on='compare_id')
    diff_train['pre_target_diff'] = oof_lgb
    diff_train['pre_target'] = diff_train['pre_target_diff'].values + diff_train['compare_target'].values

    mean_result = diff_train.groupby('样本id', sort=True)['pre_target'].mean().reset_index(name='pre_target_mean') # pd.Series.reset_index
    # true_result = train[['样本id', 'compare_target']]
    mean_result = pd.merge(mean_result, base_df, how='left', left_on='样本id', right_on='compare_id')

    print("target_diff CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, diff_target)))
    print("target CV score: {:<8.8f}".format(mean_squared_error(mean_result['pre_target_mean'].values,  mean_result['compare_target'].values)))
    
    result_train = train[['样本id']].copy()
    result_train['收率'] = mean_result['pre_target_mean'].values
    
    result_train.to_csv('./train_data/result_train_FuSai.csv', index=False, header=None)


# 还原test target
diff_test['compare_id'] = diff_test['样本id'] - diff_test['样本id_difference']
diff_test = pd.merge(diff_test, base_df, how='left', on='compare_id')
diff_test['pre_target_diff'] = predictions_lgb
diff_test['pre_target'] = diff_test['pre_target_diff'] + diff_test['compare_target']

mean_result_test = diff_test.groupby(diff_test['样本id'], sort=False)['pre_target'].mean().reset_index(name='pre_target_mean')
test = pd.merge(test, mean_result_test, how='left', on='样本id')


sub_df = pd.read_csv(TEST_PATH, encoding = 'gb18030')[['样本id']]
sub_df['收率'] = test['pre_target_mean'].values

sub_df.to_csv('./submit_%s.csv'%TEST_NAME, index=False, header=False)


