# -*- coding = utf-8 -*-
# @Time : 2022/11/4 9:17
# @Author :wangzhe
# @File : new——test.py
# @Software : PyCharm
import os
import numpy as np
#
# .txt文件的路径
# path = 'pos65000_test.txt'
#
# with open(path) as f1:
#     cNames = f1.readlines()  #.readlines()读取.txt文件的每行
#     for i in range(0,len(cNames)):
#         cNames[i] = cNames[i].strip()+' 1'+'\n'  #.strip()用于移除字符串头尾指定的字符(默认为空格或换行符）
#
# #open(path,'w')以可写方式打开.txt文件，将处理过的cNames写入新的文件中
# with open(path,'w') as f2:
#     f2.writelines(cNames)
file1 = 'neg800000.txt'
file2 = 'pos65000.txt'

def merge(file1, file2):
    f1 = open(file1, 'a+', encoding='utf-8')
    with open(file2, 'r', encoding='utf-8') as f2:
        for i in f2:
            f1.write(i)


merge(file1, file2)

