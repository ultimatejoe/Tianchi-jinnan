import pandas as pd
import numpy as np
import os
import joblib

testA = pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding='gbk')
testB = pd.read_csv('./data/jinnan_round1_testB_20190121.csv', encoding='gbk')
test = pd.read_csv('./data/jinnan_round1_test_20190201.csv', encoding='gbk')

ansA = pd.read_csv('./data/jinnan_round1_ansA_20190125.csv', encoding='gbk', header=None)
ansB = pd.read_csv('./data/jinnan_round1_ansB_20190125.csv', encoding='gbk', header=None)
ans = pd.read_csv('./data/jinnan_round1_ans_20190201.csv', encoding='gbk', header=None)

testA['收率'] = ansA[1]
testB['收率'] = ansB[1]
test['收率'] = ans[1]

train = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding='gbk')

train_new = pd.concat([train, testA, testB, test], axis=0)
train_new.reset_index(drop=True, inplace=True)
# train_new.to_csv('./data/FuSai_train.csv', index=False)


# 针对训练集中存在重复值的情况
def detect_dup(df):
    # sort df
    df_sorted = df.sort_values(by=list(df.columns[1:-1]))
    df_sorted = df_sorted.reset_index(drop=True)
    
    rest_index = df_sorted.drop_duplicates(list(df.columns[1:-1])).index
    drop_index = [idx for idx in df_sorted.index if idx not in rest_index]
    
    return df_sorted, rest_index, drop_index

def avg(df, start, end):
    df.loc[start-1, '收率'] = np.average(df.loc[start-1:end, '收率'])
    

def seg_avg(df, a):
    '''
    avg for dups
    '''
    start, i, N = 0, 0, len(a)
    if N == 0:
        return 
    while i<N:
        if (a[i] - a[start]) == (i - start):
            i += 1            
        else:
            avg(df, a[start], a[i-1])
            start = i
    avg(df, a[start], a[i-1])


df_sorted, rest_index, drop_index = detect_dup(train_new)
seg_avg(df_sorted, drop_index)
result_df = df_sorted.loc[rest_index]

result_df.reset_index(drop=True, inplace=True)
result_df.to_csv('./data/FuSai_train.csv', index=False)