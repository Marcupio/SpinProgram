# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:58:09 2017

@author: nac2313
"""

import numpy as np
import Hamiltonians as H
import FilteredStates as fs
import VNEntropy as VN
import Expand as Ex
import os
import pathlib as pl
import multiprocessing as mp
from matplotlib import pyplot as plt
import time

#Older Code Use EECalc instead
def DisVEntOld(N,Nlegs,Na,Nb,phiamt,Nstates=30,Jcurr=0,J1=1,J2=0,gamamt=50, modex2='open', modey2='open'):

    '''
    N = 16
    Nstates = 30
    Jcurr = 0
    phiamt = 100
    gamamt = 50
    Nlegs = 8
    Na = 8
    Nb = 8
    J1=1
    J2=1
    '''
    #print('here')
    S = 0.5
    c = np.sqrt(2)
    
    phis = np.linspace(-np.pi*0.5,np.pi*0.5,phiamt)
    gammas = np.linspace(0.5,8,gamamt)
    betas = [c,0]
    NAsites = [0,1,2,3,4,5,6,7]
    NBsites = [8,9,10,11,12,13,14,15]
    
    SA = []
    for gamma1 in gammas:
        
        Ent1 = []
        start1 = time.time()
        for phi1 in phis:
            
            Hfull = H.nlegHeisenberg.blockH(N,S,Nlegs,c,Js=[J1,J2],gamma=[0,gamma1],phi=[0,phi1], modex1=modex2, modey1=modey2)#H.TwoDHeisenberg.H(N,Nlegs,S,J = [-1,0],delx=del1,dely=del2,delz=del3,beta=betas,gamma=[gammas[i],0],phi = [phis[j],0])#H.OneDHeisenberg.H(N,S,J1=-1,gamma=gammas[i],beta=c,phi=phis[j], mode='open')### Generate Hamiltonian
            Htot = sum(Hfull)
            #end1 = time.time()
            #print('Hamiltonian Generateion time:', end1-start1)
            
            #H1 = rows@Htot@rows.T#Reduce to Sz=Jcurr Hamiltonian
            #start = time.time()
            Energy, EnergyStates = fs.filterHstates(Htot,Nstates)#filter states close to energy density
            #Entropies = []
            
            for EnergyState in EnergyStates:
                Exstate = Ex.Expand(EnergyState,S,N,Jcurr)#expand reduced-hamiltonian state to hilbert space of full hamiltonian
                #print(Exstate.tolist())
                #RhoA = VN.reducedDM(EnergyState,NAsites,NBsites,N,S,Jcurr=Jcurr)
                RhoA = VN.bipartDM(Exstate,Na,Nb)
                #print(RhoA)
                eigsA = np.linalg.eigvals(RhoA)
                
                Ent1.append(VN.VNEnt(eigsA))
            
            
        SA.append(np.mean(Ent1).real)
        end = time.time()
        print('Time for one phi:',end-start1)
        #print(np.mean(Ent1))
        #SA.append(np.mean(Ent1))
    
    str1 = pl.PurePath(modex2+'x_'+modey2+'y')
    path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Entropy'  ###path and filepaths are objects which allow 
    print(str(path))
    filename = '%dx%d_%dphis_%dNstates_J1_%d__J2_%d__%dgammas_Na_%d__Nb_%d__%sx_%sy' % (N/Nlegs,Nlegs,phiamt,Nstates,J1,J2,gamamt,Na,Nb,modex2,modey2)
    filepath = path/filename
    np.save(str(filepath),SA)

def DisVEnt(N,Nlegs,Na,Nb,phiamt,Nstates=30,Jcurr=0,J1=1,J2=1,gamamt=50, modex2='open', modey2='open'):
    
    S = 0.5
    c = np.sqrt(2)
    
    phis = np.linspace(-np.pi*0.5,np.pi*0.5,phiamt)
    gammas = np.linspace(0.5,8,gamamt)
    #betas = [c,0]
    #NAsites = [0,1,2,3,4,5,6,7]
    #NBsites = [8,9,10,11,12,13,14,15]
    SA = []
    
    
    for gamma1 in gammas:
    
        Ent1 = []
    
        start1 = time.time()
        def getH(phi):
        
            Hfull = H.nlegHeisenberg.blockH(N,S,Nlegs,c,Js=[J1,J2],gamma=[gamma1,0],phi=[phi,0], modex1=modex2, modey1=modey2)
            return(sum(Hfull))
    
        def getEntropies(Htot):
            Energy, EnergyStates = fs.filterHstates(Htot,Nstates)#filter states close to energy density
        #Entropies = []
            def getEnt(EnergyState):
                Exstate = Ex.Expand(EnergyState,S,N,Jcurr)
                RhoA = VN.bipartDM(Exstate,Na,Nb)
                eigsA = np.linalg.eigvals(RhoA)
                return(VN.VNEnt(eigsA))
            return(np.mean(list(map(getEnt,EnergyStates))))
        
        Hamiltonians = map(getH,phis) 
        Ent1 = list(map(getEntropies,Hamiltonians))#this list is as long as the amount of phis, one entry per phi <S>
            
        SA.append(np.mean(Ent1).real)#this is an average over phi [<S>]
        end1= time.time()
        print('Time for one gamma:',end1-start1)
    
    str1 = pl.PurePath(modex2+'x_'+modey2+'y')
    path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Entropy' ###path and filepaths are objects which allow  
    filename = '%dx%d_%dphis_%dNstates_J1_%d__J2_%d__%dgammas_Na_%d__Nb_%d__%sx_%sy' % (N/Nlegs,Nlegs,phiamt,Nstates,J1,J2,gamamt,Na,Nb,modex2,modey2)
    filepath = path/filename
    np.save(str(filepath),SA)
    #return(SA)
