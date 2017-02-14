#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:01:29 2017

@author: zhangchi
"""

import pandas as pd
fout = open('result0213.txt','w')
fileNames = ['Bed006', 'Bed012', 'Bed016', 'Bmr001', 'Bmr010', 'Bmr022', 'Bmr028', 'Bro008', 'Bro014', 'Bro021', 'Bro027','Bed003', 'Bed010', 'Bmr005', 'Bmr014', 'Bmr019', 'Bmr024', 'Bmr030', 'Bro004', 'Bro011', 'Bro018', 'Bro024','Bdb001', 'Bed002', 'Bed004', 'Bed005', 'Bed008', 'Bed009', 'Bed011', 'Bed013', 'Bed014', 'Bed015', 'Bed017', 'Bmr002', 'Bmr003', 'Bmr006', 'Bmr007', 'Bmr008', 'Bmr009', 'Bmr011', 'Bmr012', 'Bmr015', 'Bmr016', 'Bmr020', 'Bmr021', 'Bmr023', 'Bmr025', 'Bmr026', 'Bmr027', 'Bmr029', 'Bmr031', 'Bns001', 'Bns002', 'Bns003', 'Bro003', 'Bro005', 'Bro007', 'Bro010', 'Bro012', 'Bro013', 'Bro015', 'Bro016', 'Bro017', 'Bro019', 'Bro022', 'Bro023', 'Bro025', 'Bro026', 'Bro028', 'Bsr001', 'Btr001', 'Btr002', 'Buw001']
for fileName in fileNames:
    print fileName
    da = pd.DataFrame.as_matrix(pd.read_csv('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/data/'+fileName+'.dadb',usecols=[5]))
    sentence = pd.DataFrame.as_matrix(pd.read_csv('/Users/zhangchi/Desktop/cs690/icsi_mrda+hs_corpus_050512/data/'+fileName+'.trans',usecols=[1]))
    for i in range(len(da)):
        try:
            if 'qw' in da[i][0]:
            #if 'qw' == da[i][0]:
                fout.write('qw: '+sentence[i][0]+'\n')
        except:
            pass
fout.close()