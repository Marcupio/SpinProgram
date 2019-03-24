# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:46:09 2018

@author: nac2313
"""

'''
Calculates Biparte Entanglement Entropy, and Adjacent Gap Ratio for a given Hamiltonian.

'''
import numpy as np
import Hamiltonians as H
import FilteredStates as fs
import VNEntropy as VN
import Expand as Ex
import os
import pathlib as pl
import time



def PhaseAttr(N,Nlegs,Na,Nb,gamval,phiamt,Nstates=30,Jcurr=0,J1=1,J2=1, modex2='open', modey2='open'):
    
    #change of plans, we calculate only single values for a given gamma better for multiprocessing since all gammas are independent of one another
    S = 0.5
    c = np.sqrt(2)
    
    phis = np.linspace(-np.pi*0.5,np.pi*0.5,phiamt)
    #betas = [c,0]
    #NAsites = [0,1,2,3,4,5,6,7]
    #NBsites = [8,9,10,11,12,13,14,15]
    
    def getH(phi):
        
        Hfull = H.nlegHeisenberg.blockH(N,S,Nlegs,c,Js=[J1,J2],gamma=[gamval,0],phi=[phi,0], modex1=modex2, modey1=modey2)
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
        
    start1 = time.time()
    Hamiltonians = list(map(getH,phis)) 
    Stats = list(map(getStats,Hamiltonians))#returns a 2D array, the first row is a list of Entropies, the second is a list of AGRs 
        
        
    Ent1=[Stat[0] for Stat in Stats]
    AGR1 = [Stat[1] for Stat in Stats]
        
    #each list has phi amt of elements, and so it averaged over giving us [<S>],[<AGR>] for a given gamma
    SA = np.mean(Ent1)
    AGR = np.mean(AGR1)
        
    end1= time.time()
    print('time for one gamma: ',end1-start1)
    
    return((SA,AGR))

