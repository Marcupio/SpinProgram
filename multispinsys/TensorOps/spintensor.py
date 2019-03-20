# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:51:56 2017

@author: crius
"""

import scipy as sp
from scipy import sparse


def spintensor(k,N,s,S):
    '''
        Gives spintensor of individual spin expanded to the full hilbert space of N spins e.g. S1
    '''
    A = sp.sparse.kron(sp.sparse.identity((2*S+1)**(k-1)), sp.sparse.kron(s,sp.sparse.identity((2*S+1)**(N-k))))

    return(A)

def fullspintensor(N,s,S):
    '''
        Gives total spintensor Sz(Sx/Sy) for N spins e.g. S1 + S2 +...+Sn
    '''
    SpT = []
    for k in range(1,N+1):
        SpT.append(spintensor(k, N, s,S))
        
    return(sum(SpT))

def parspintensor(Nsites,N,s,S):
    '''
        Gives  total spintensor Sz(Sx/Sy) of first Nsites for a collection of N spins.
    '''
    SpT = []
    print()
    for k in Nsites:
        SpT.append(spintensor(k,N,s,S))
    
    return(sum(SpT))


    