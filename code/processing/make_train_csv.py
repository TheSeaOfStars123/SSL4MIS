'''
  @ Date: 2022/3/24 10:28
  @ Author: Zhao YaChen
'''
# @Time : 2021/9/18 11:08 AM
# @Author : zyc
# @File : make_train_csv.py
# @Title : 根据name_mapping.csv和MR_list.xls文件生成train_csv文件
# @Description : 将之间生成的两个文件进行合并，并生成图像路径新的一列
from config import opt
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

MR_list_df = pd.read_csv(opt.MR_list_path, encoding = 'unicode_escape')
name_mapping_df = pd.read_csv(opt.name_mapping_path)

MR_list_df.rename({'ID': 'Number'}, axis=1, inplace=True)
df = MR_list_df.merge(name_mapping_df, on='Number', how="right")

# 生成图像路径新的一列
paths = []
for _, row in df.iterrows():

    id_ = row['Breast_subject_ID']  # id_ = Breast_Training_00X
    phase = id_.split("_")[-2]

    if phase == 'Training':
        path = os.path.join(opt.train_root_dir, id_)
    else:
        path = os.path.join(opt.test_root_dir, id_)
    paths.append(path)

df['path'] = paths

#split data on train, test, split

train_data = df.loc[df['PCR'].notnull()].reset_index(drop=True)
train_data["PCR_rank"] =  train_data['PCR'] // 1
# train_data = train_data.loc[train_data['Breast_subject_ID'] != 'Breast_Training_308'].reset_index(drop=True, )

# StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
skf = StratifiedKFold(
    n_splits=7, random_state=opt.seed, shuffle=True
)
for i, (train_index, val_index) in enumerate(
        skf.split(train_data, train_data["PCR_rank"])
        ):
        train_data.loc[val_index, "fold"] = i

train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)

test_df = df.loc[~df['PCR'].notnull()].reset_index(drop=True)
print("train_df ->", train_df.shape, "val_df ->", val_df.shape, "test_df ->", test_df.shape)
train_data.to_csv(opt.path_to_csv, index=False)

