import os
os.system("python train_data.py") # 合并之前测试集和训练集，形成新的训练集
os.system("python model_FuSai.py") 
os.system("python model_optimize.py")