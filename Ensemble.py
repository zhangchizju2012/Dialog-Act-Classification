#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 23:11:44 2017

@author: zhangchi
"""

target = np.array(targetList_dev_CNN)
sco_cnn = np.array(scoresList_dev_CNN)
sco_rnn = np.array(scoresList_dev_RNN)
sco_para = np.array(scoresList_dev_ParaRNNCNN)
sco_rnncnn = np.array(scoresList_dev_CNNRNN)
a = 0.3 * sco_cnn + 0.5 * sco_para +sco_rnn + sco_rnncnn
count = 0
for i in range(39):
    count = count + np.sum(np.argmax(a[i],axis=1)==target[i])
    
b=target.reshape(39*64)
c=np.argmax(a,axis=2)
d=c.reshape(39*64)

for i in range(8):
    print np.sum(b==i)

result = [[0]*8,[0]*8,[0]*8,[0]*8,[0]*8,[0]*8,[0]*8,[0]*8]
for i in range(39*64):
    result[b[i]][d[i]] = result[b[i]][d[i]] + 1