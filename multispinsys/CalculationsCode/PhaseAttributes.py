# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:55:43 2018

@author: nac2313
"""
'''
Calculates Biparte Entanglement Entropy, and Adjacent Gap Ratio for a given Hamiltonian.

'''
import numpy as np
import random
import Hamiltonians as H
import FilteredStates as fs
import VNEntropy as VN
import Expand as Ex
import os
import pathlib as pl
import time

##Old Code used to calculate EE and AGR

def PhaseAttr(N,Nlegs,Na,Nb,phiamt,c=np.sqrt(2),Nstates=30,Jcurr=0,J1=1,J2=1,ratio=[1,1], gamamt=10, modex2='open', modey2='open'):
    #ratio is used to adujust disorder strengths along directions i.e. gamma[1] = ratio*gamma1 this allows easy manipulation of relative disorder strengths between directions
    
    S = 0.5
    
    
    def getrand():
        return(random.uniform(0,np.pi))


    gammas = np.linspace(0.5,8,gamamt)
    ratio1=ratio[0]
    ratio2=ratio[1]

    SA = []
    AGR = []
    
    for i,gamma1 in enumerate(gammas):
        
       
        def getH(phi):
            
            Hfull = H.nlegHeisenberg.blockH(N,S,Nlegs,c,Js=[J1,J2],gamma=[ratio1*gamma1,ratio2*gamma1],phi=[phi[0],phi[1]], modex1=modex2, modey1=modey2)
            return(sum(Hfull))
    
        def getStats(Htot):
              
            def getEnt(EnergyState):#finds entanglement entropy for a given state of H(phi)
            
                Exstate = Ex.Expand(EnergyState,S,N,Jcurr)
                RhoA = VN.bipartDM(Exstate,Na,Nb)
                eigsA = np.linalg.eigvals(RhoA)
                
                return(np.mean(VN.VNEnt(eigsA)).real)
            
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
            
            Energy, EnergyStates = fs.filterHstates(Htot,Nstates)#filter states close to energy density
            sortEn = sorted(Energy)
            
            Entropies = map(getEnt,EnergyStates)
            AGRs = getAGR(sortEn)
            
            return(list(Entropies), list(AGRs))
            #return(np.mean(list(map(getEnt,EnergyStates))))
        #phis = [(getrand(),getrand()) for i in range(phiamt)]
        phis=[(np.pi/12,np.pi/12)]
        start1 = time.time()
        Hamiltonians = list(map(getH,phis))
        Stats = list(map(getStats,Hamiltonians))#returns a 2D array, the first row is a list of Entropies, the second is a list of AGRs 
        
        
        Ent1=[Stat[0] for Stat in Stats]
        AGR1 = [Stat[1] for Stat in Stats]
        
        #each list has phi amt of elements, and so it averaged over giving us [<S>],[<AGR>] for each gamma
        SA.append(np.mean(Ent1))
        AGR.append(np.mean(AGR1))
        
       
        end1= time.time()
        print('time for gamma index: %d' %(i),end1-start1)
    str1 = pl.PurePath(modex2+'x_'+modey2+'y')
    c1 = str(round(c,3))
    
    #saves data for Entropy
    path1 = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Entropy' ###path and filepaths are objects which allow  
    filename1 = '%dx%d_%dphis_%dNstates_J1_%d__J2_%d__%dgammas_Na_%d__Nb_%d__%sx_%sy_disorder_onX_onY__c%s__ratio_%d_%d' % (N/Nlegs,Nlegs,phiamt,Nstates,J1,J2,gamamt,Na,Nb,modex2,modey2,c1,ratio1,ratio2)
    filepath1 = path1/filename1
    #np.save(str(filepath1),SA)
    
    #saves data for AGR
    path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'AGR' ###path and filepaths are objects which allow  
    filename = 'AGR_%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%sx_%sy_disorder_onX_onY__c%s__ratio_%d_%d' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt,modex2,modey2,c1,ratio1,ratio2)
    filepath = path/filename
    #np.save(str(filepath),AGR)
    return(SA,AGR)
'''
N=16
Na=8
Nb=8
phiamt=1
Nlegs=4
c2=np.sqrt(2)

stats = PhaseAttr(N,Nlegs,Na,Nb,phiamt,c=c2,gamamt=10)
'''

