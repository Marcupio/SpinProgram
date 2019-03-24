# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:55:16 2018

@author: nac2313
"""
import os
import random
import pathlib as pl
import Hamiltonians as H
import FilteredStates as fs
import numpy as np
from matplotlib import pyplot as plt
import time

#used to find AGRs for a range of gammas

def AGR(N,Nlegs,phiamt,c=np.sqrt(2),Nstates=30,Jcurr=0,J1=1,J2=1,ratio=[1,1], gamamt=10, gammax=10, modex2='open', modey2='open'):
    
    S=0.5
    #phi = np.linspace(-np.pi*0.5,np.pi*0.5,phiamt)
    def getrand():
        return(random.uniform(0,np.pi))

    gammas = np.linspace(0.5,gammax,gamamt)
    ratio1=ratio[0]
    ratio2=ratio[1]

    
    AGRs = []
    
    for i,gamma1 in enumerate(gammas):
        
       
        def getH(phi):
            
            Hfull = H.nlegHeisenberg.blockH(N,S,Nlegs,c,Js=[J1,J2],gamma=[ratio1*gamma1,ratio2*gamma1],phi=[phi[0],phi[1]], modex1=modex2, modey1=modey2)
            return(sum(Hfull))
    
        def getAGRs(Htot):
            
            def getAGR(sortedEn):
                
                Adgphi = []#find all adgr(r) for a given phi, but each phi corresponds to a unique Hamiltonian i.e. H(phi)
                for n in range(1,len(sortedEn)-1):
                
                    Enb = sortEn[n+1]
                    Ena = sortEn[n-1]
                    En = sortEn[n]
                    AGn = Enb - En
                    AGa = En - Ena
                    
                    A = min(AGn,AGa)/max(AGn,AGa)
                    Adgphi.append(A)
                    
                return(Adgphi)
            
            Energy= fs.filterEnVals(Htot,Nstates)#filter states close to energy density
            sortEn = sorted(Energy)
            AGR1 = getAGR(sortEn)
            
            return(list(AGR1))
        
        start1 = time.time()
        phis = [(getrand(),getrand()) for i in range(phiamt)]
        Hamiltonians = list(map(getH,phis))
        AGR2 = list(map(getAGRs,Hamiltonians)) 
        
        #each list has phi amt of elements, and so it averaged over giving us [<S>],[<AGR>] for each gamma
        AGRs.append(np.mean(AGR2))
        end1= time.time()
        print('time for gamma index: %d' %(i),end1-start1)
        
    #generates file names and paths
    str1 = pl.PurePath(modex2+'x_'+modey2+'y')
    c1 = str(round(c,3))#rounds the value of c out to 3 decimal places and converts it to string to be used in file destination
    
    #saves data for AGR
    path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'AGR' ###path and filepaths are objects which allow the program to locate data
    filename = 'AGR_%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%dgammax_%sx_%sy_disorder_onX_onY__c%s' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt,gammax,modex2,modey2,c1)
    filepath = path/filename
    #np.save(str(filepath),AGRs)
    #return(AGRs)
