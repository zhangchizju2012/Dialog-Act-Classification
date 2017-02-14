#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:01:29 2017

@author: zhangchi
"""

import pandas as pd
import os

fout = open('result0213.txt','w')

for name in os.listdir('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/reliability/hot_spots/data_chunks/'):
    eachFile = pd.read_csv('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/reliability/hot_spots/data_chunks/'+name,usecols=[3,7])
    #eachFile = pd.read_csv('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/reliability/hot_spots/data_chunks/Bed003.600-1200.RD.csv',usecols=[3,7])
    eachFile = pd.DataFrame.as_matrix(eachFile)
    for i in range(len(eachFile)):
        try:
            if 'qw' in eachFile[i][0]:
                fout.write('qw: '+eachFile[i][1]+'\n')
        except:
            pass

for name in os.listdir('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/reliability/dialog_acts/data_chunks/'):
    eachFile = pd.read_csv('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/reliability/dialog_acts/data_chunks/'+name,usecols=[3,6])
    #eachFile = pd.read_csv('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/reliability/hot_spots/data_chunks/Bed003.600-1200.RD.csv',usecols=[3,7])
    eachFile = pd.DataFrame.as_matrix(eachFile)
    for i in range(len(eachFile)):
        try:
            if 'qw' in eachFile[i][0]:
                fout.write('qw: '+eachFile[i][1]+'\n')
        except:
            pass
fout.close()