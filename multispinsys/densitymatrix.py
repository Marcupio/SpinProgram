# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 01:01:27 2017

@author: nac2313
"""

import numpy as np
import Hamiltonians as H
import scipy as sp
import FilteredStates as fs
import time
def densitymatrixB(State,eigs='False'):

    #EigsB, StatesB = np.linalg.eigh(HB)
   
   
    #rows = np.transpose(StatesB)
    
    #EigsRhoA = [np.asscalar(np.conj(np.dot(row,StatesT)*np.dot(row,StatesT))) for row in rows]# for i in range (0,StatesT.shape[0])]
   
    #return(EigsRhoA)
           #eig = np.asscalar(np.dot(row,StatesT))
           #EigsRhoA.append(np.conjugate(eig)*eig)# for row in rows]
    #states = []
    #print(SzAvecs)
    #for colA in SzAvecs:
    #    for colB in SzBvecs:
            
    #        states.append(np.kron(colA,colB))
    
    #dim = len(states)
    #row = np.transpose(StateT)
    
    #RhoT = [np.asscalar(np.vdot(row,state)) for state in states]
    #print(RhoT)
    #print(RhoT)
    
    
    #M = np.reshape(RhoT,(int(len(RhoT)/2),2))
    row = np.conjugate(np.transpose(State))
    
    RhoB = np.kron(State,row)
    print(RhoB)
    if eigs == 'True':
        
        eigsB = np.linalg.eigvals(RhoB)
        
        return(eigsB)
    
    elif eigs == 'False':
        
        return(RhoB)

    
    
    