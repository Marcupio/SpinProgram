# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 00:18:33 2018

@author: nac2313
"""

import Hamiltonians as H
import numpy as np
import scipy as sp
import spinops as so
import spintensor as st
import time

N = 4
S = 0.5
Jcurr = 0
sz = so.sz(S)
Na = 1
Nb = N-Na
NAsites = np.arange(1,Na+1)

#generate non-block diagonal Hamiltonian
H = H.OneDHeisenberg.H(N,S,mode='open')
Hfull = sum(H).todense()
Eigs, Evecs = np.linalg.eigh(Hfull)
Estates = Evecs.T

stateMat = np.kron(state.T,state)

SzpTA = st.parspintensor(NAsites,Na,sz,S)

eigs, vecs = np.linalg.eigh(SzpTA.todense())
Sstates = vecs.T

dim = int((2*S+1)**Nb)
Exstates = []

for row in Sstates:

    Exrow = np.kron(row,np.eye(dim))
    Exstates.append(Exrow)
   
rho = []  
for StateA in Exstates:
    
    rho.append(StateA@stateMat@StateA.T)

rhoA = sum(rho)
print(rhoA)


