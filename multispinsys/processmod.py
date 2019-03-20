# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:45:36 2018

@author: nac2313
"""
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import PhaseAttributes as pa
import AdjacentGapRatioTest as agrT
import DisorderVEntropy as ent
import pathlib as pl
import os
import time

N = 12
o = 'open'
p = 'periodic'
modex=p
modey=o
gamamt1=20
gammax1 = 12
c1=0
params = np.linspace(0.5,gammax1,gamamt1)

Nstates=30
J1=1
J2=1
phiamt=1000
Nlegs = 2
Na=8
Nb=8

def getprocess(param1):
    #returns SA,AGR per gamma as a tuple (SA,AGR)
    return(agrT.AGR(N,Nlegs,phiamt,param1,ratio=[1,0],c=c1,modex2=modex,modey2=modey))#it can be any function you want just need appropriate parameters(param)
    
    
if __name__=='__main__':
    start=time.time()
    p = Pool(15)
    #params1=[params[i] for i in range(5)]
    data = list(p.map(getprocess,params))
    print(data)
    end=time.time()
    print('time: ',end-start)
    
    str1 = pl.PurePath(modex+'x_'+modey+'y')
    c2 = str(round(c1,3))#rounds the value of c out to 3 decimal places and converts it to string to be used in file destination
    
    #saves data for AGR
    path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'AGR' ###path and filepaths are objects which allow the program to locate data
    filename = 'AGR_%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%dgammax_%sx_%sy_disorder_onX_onY__c%s__r_1_0' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt1,gammax1,modex,modey,c2)
    filepath = path/filename
    np.save(str(filepath),data)
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
    
