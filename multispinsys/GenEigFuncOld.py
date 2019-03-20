# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:38:28 2019

@author: nac2313
"""

import numpy as np
import Hamiltonians as H
import time
import multiprocessing as mp
import random
from multiprocessing import Pool
import os
import pathlib as pl
import FilteredStates as fs

#This code will generate, filter, and store eigenvalues and eigenfuctions for Hamiltonians as a function of some parameter, in this case disorder(gamma). 
# for an odd number of spins, the total SZ is set to 0.5
N = 8
o = 'open'
p = 'periodic'
modex=o
modey=o
gamamt1=1
gammax1 =30
c=np.sqrt(2)
uniform1='False'
Sztotal = 0

gammas = np.linspace(0.5,gammax1,gamamt1)

Nstates=30
J1=1
J2=1
phiamt=10
Nlegs = 3


c2 = str(round(c,3))#rounds the value of c out to 3 decimal places and converts it to string to be used in file destination
filename = 'Eigfuncs__%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%dgammax__%dNstates__%sx_%sy_disorder_onX_onY__c%s__uniform_%s' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt1,gammax1,Nstates,modex,modey,c2,uniform1)
print(filename)#used as a double check to ensure that all parameters are correct; all pertainent info is in the filename.

def getrand():
    return(random.uniform(0,np.pi))

def getHams(gamma1):
    
    def getH(phi):
         
        Hfull = H.nlegHeisenberg.blockH(N,Nlegs,c,TotalSz=Sztotal,Js=[J1,J2],gamma=[gamma1,gamma1],phi=[phi[0],phi[1]], modex1=modex, modey1=modey,uniform=uniform1)
        
        return(sum(Hfull))
    
    phis = [(getrand(),getrand()) for i in range(phiamt)]
    Hams1 = list(map(getH,phis))
    
    
    return(Hams1)

def getEvecsvals(Htot):
        
    vals, vecs = fs.filterHstates(Htot,Nstates)#gets closest Nstates to given energy density (enden). enden is default 0.5
    return(vals,vecs)

Ham_matri= list(map(getHams, gammas))#for each gamma it calculates the Hamiltonians as a function of phi: H(phi). Each element of Ham_matri is a list of phiamt matrices for the gamma value

if __name__=='__main__':
    start=time.time()
    
    Eigfuncs = []
    
    for Hams2 in Ham_matri:#for each gamma value in our matrix of Hamiltonians we have a list of Hamiltonians
        
        start = time.time()
        processes = []
        for h1 in Hams2:#for each Hamiltonian in the list we diagonalize
            
            p = mp.Process(target=getEvecsvals, args=(h1,))
            processes.append(p)
            p.start()
        
        for process in processes:
            process.join()
       
        Eigfuncs.append(data)
        end=time.time()
        print('Time for one h:  ',end-start)
    print(Eigfuncs[0])
    Evecs = [[Eigfuncs[g][1][i][1] for i in range(phiamt)]for g in range(gamamt1)]
    Evals = [[Eigfuncs[g][1][i][0]for i in range(phiamt)]for g in range(gamamt1)]
    
    #print(Evecs)
    
    end=time.time()
    #print('total time: ',end-start)
    
    #saves data for Eigfuncs
    str1 = pl.PurePath(modex+'x_'+modey+'y')#sorts by boundary conditions first
    
    path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Eigenfunctions' ###path and filepaths are objects which allow the program to locate data
    
    filename = 'Eigfuncs__%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%dgammax__%dNstates__%sx_%sy_disorder_onX_onY__c%s__uniform_%s' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt1,gammax1,Nstates,modex,modey,c2,uniform1)
    filepath = path/filename
    
    
    #Eigfuncs['Evals','Evecs']
    #Each Element (row) in Evals, and Evecs, corresponds to a single gamma from the array np.linspace(0.5,gammax,gamamt). 
    #Each element in a given gamma-row corresponds to the list of eigenvalues/vectors for a single (phi1,phi2) pair used by the Hamiltonian module.
    #e.g. Eigfuncsfile['Evals'][0] gives the set of eigenvalue sets for a certain gamma, and Eigfuncsfile['Evals'][0][0] gives the set of eigenvalues pertaining to a single phi-pair, for a certain gamma.
   
    #np.savez(str(filepath),Evecs=Evecs,Evals=Evals)#saves the pair of arrays as filname['Evecs','Evals']