'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/1/16 21:15
@Author : Qiufen.Chen
@FileName: split_data.py
@Software: PyCharm
'''

import pandas as pd
import numpy as np

import os

currentPath = os.getcwd()  # 
parent_path = os.path.dirname(currentPath)  # 
print(parent_path)

data_path = parent_path + '/data/original_data/classification_data.csv'
data = pd.read_csv(data_path, usecols=[0, 1, 2])

array = np.array(data)
li = array.tolist()

ratio = 2
positive = 480
negtive = ratio * positive

res = li[:negtive]
name = ['ID', 'SMILES', 'AV_Bit']
test = pd.DataFrame(columns=name, data=res)  # one,two,three
test.to_csv(parent_path + '/data/1-1_data.csv', encoding='utf-8')



