# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 22:16:15 2019

@author: nac2313
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import PhaseAttributes as pa
import AdjacentGapRatio as agr
import DisorderVEntropy as ent
import pathlib as pl
import os
import time

N = 8
o = 'open'
p = 'periodic'
modex=o
modey=o
gamamt1=20
gammax=30
params = [0]#np.linspace(0.5,8,gamamt)
Nstates=30
J1=1
J2=1
phiamt=10
Nlegs = 2
Na=8
Nb=8

def getprocess(param1):
    #returns SA,AGR per gamma as a tuple (SA,AGR)
    return(agr.AGR(N,Nlegs,phiamt,c=param1,gamamt=gamamt1))#it can be any function you want just need appropriate parameters(param)
    
    
if __name__=='__main__':
    start=time.time()
    p = Pool(5)
    p.map(getprocess,params)
    end=time.time()
    print('time: ',end-start)
    '''
    SA = [s[0] for s in Stats]
    AGR = [a[1] for a in Stats]

    

    str1 = pl.PurePath(modex+'x_'+modey+'y')
    
    #saves data for Entropy
    path1 = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Entropy' ###path and filepaths are objects which allow  
    filename1 = '%dx%d_%dphis_%dNstates_J1_%d__J2_%d__%dgammas_Na_%d__Nb_%d__%sx_%sy_disorder_onX_onY__c%s' % (N/Nlegs,Nlegs,phiamt,Nstates,J1,J2,gamamt,Na,Nb,modex,modey)
    filepath1 = path1/filename1
    #np.save(str(filepath1),SA)
    
    #saves data for AGR
    path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'AGR' ###path and filepaths are objects which allow  
    filename = 'AGR_%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%sx_%sy_disorder_onX_onY__c%s' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt,modex,modey)
    filepath = path/filename
    #np.save(str(filepath),AGR)
    '''
    