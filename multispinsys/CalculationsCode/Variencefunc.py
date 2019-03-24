# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:35:57 2018

@author: crius
"""
import time
import FilteredStates as fs
import tools as to
import itertools as it
import Hamiltonians as H
import numpy as np
from matplotlib import pyplot as plt
'''
Find variance function <Sz^2>- <Sz>^2 where Sz is the total Sz operator for
system A .
'''
N=6
S=0.5
Nstates = 10
nlegs = 3
c = np.sqrt(2)
Na=3

Slist = to.Statelist(N,S)
ms = []
for i,state in enumerate(Slist):
    mstate = 0
    for spin in it.islice(state,Na):
        
        if spin == 0:
            m = -1
        elif spin == 1:
            m = 1
        
        mstate = mstate + m
        
    ms.append((i,mstate))

mdict = dict(ms)
statedict = dict([(i,state) for i,state in enumerate(Slist)])

def term1(envec):
    ter1 = sum([(mdict[i]**2)*abs(coeff)**2 for i,coeff in enumerate(envec)])
    return(ter1)

def term2(envec):
    ter2 = sum([mdict[i]*abs(coeff)**2 for i,coeff in enumerate(envec)])
    return(ter2)

def avgVar(envecs):
    
    aV = np.mean([0.25*term1(envec)-term2(envec)**2 for envec in envecs])
    
    return(aV)

phis = np.linspace(-np.pi*0.5,np.pi*0.5,10)

gammas = np.linspace(0,8,50)
Vars = []
for gamma1 in gammas:
    Var = []
    start= time.time()
    for phi1 in phis:
       
        Htot = sum(H.nlegHeisenberg.blockH(N,S,nlegs,c,gamma=[0,gamma1],phi=[0,phi1])).todense()
        eneigs, envecs = fs.filterHstates(Htot,Nstates)
        Var.append(avgVar(envecs))
    
    Vars.append(np.mean(Var))
    end = time.time()
    print(end-start)
    
plt.plot(gammas,Vars)
plt.show()
