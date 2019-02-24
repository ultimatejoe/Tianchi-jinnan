import pandas as pd
import numpy as np
import os
import time
import joblib
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from model_evaluate import pipeline

keys = ['A1', 'A2', 'A3', 'A4',
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

prior_data = pd.read_csv('./pipeline_data/prior_data.csv')
prior_target = pd.read_csv('./train_data/result_train_FuSai.csv', header=None)[1]

prior_data = pd.read_csv('./pipeline_data/prior_data.csv')
prior_target = pd.read_csv('./train_data/result_train_FuSai.csv', header=None)[1]

prior = pd.concat([prior_data, prior_target], axis=1)
prior.drop_duplicates(subset=prior_data.columns, inplace=True)
prior = prior.dropna() # 


# 过滤反应时长超过8小时的
prior = prior.loc[prior['A20_interval']<8]
prior = prior.loc[prior['A28_interval']<8]
prior = prior.loc[prior['B4_interval']<8]

prior = prior.loc[prior['B9_B10_B11_interval']<8]

prior.reset_index(drop=True, inplace=True)
prior_data,prior_target = prior.drop(columns=[1]), prior[1]

# 输出参数范围
params_range = pd.DataFrame({'keys': keys, 
              'min':prior_data.min(axis=0).values, 
              'max':prior_data.max(axis=0).values })
range_dict = {}
for i in range(params_range.shape[0]):
    tmp = params_range.loc[i]
    range_dict[tmp['keys']] = (tmp['min'], tmp['max'])

# initialize Optimizer
optimizer = BayesianOptimization(
    f=None,
    pbounds=range_dict,
    verbose=2,
    random_state=1
)

# add prior_data
X = prior_data.values
Y = prior_target.values

cnt = 0
for x,y in zip(X, Y):
    cnt += 1
    optimizer.register(params=x,
                      target=-np.abs(1-y))  # 优化目标是收率尽可能接近1

# initialize utility function
from bayes_opt import UtilityFunction
utility = UtilityFunction(kind='ucb', kappa=2.5, xi=0)

# 搜索1000个点
for i in tqdm(range(1,1000)):
    next_point = optimizer.suggest(utility)
    tmp = pd.DataFrame(next_point, index=[0])
    target = pipeline(tmp)[0]
    optimizer.register(params=tmp[keys].values[0],
                      target=-np.abs(1-target))
    if i%10 == 0:
        joblib.dump(optimizer,'./model/optimize_model.pkl')