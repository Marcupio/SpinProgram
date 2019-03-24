# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:46:50 2017

@author: nac2313
"""
import numpy as np
import scipy as sp
import itertools
import time
from scipy.sparse.linalg import eigsh
"""
    Filters Nstates for a given energy density, for a given system H
    default: enden=0.5
"""

def filterHstates(H, Nstates, enden=0.5, Full='False'):

    #E, vecs = #np.linalg.eigh(H)
    
    if Full =='True':
        
        Eigs, vecs = np.linalg.eigh(H.todense())
        Evecs = np.asarray(vecs.conj().T)
        
        return (Eigs,Evecs)
    
    elif Full == 'False':
        
        #E = np.linalg.eigvalsh(H.todense())
        def ExmEigs(param):
            E, v = sp.sparse.linalg.eigsh(H,k=1,which=param,tol=1e-8)#finds a single eigenvalue, in this case it's either the largest or smallest based on params
            return(np.asscalar(E))
        params = ['SA','LA']
        Erange= list(map(ExmEigs,params))
        
        
        target_E = enden*Erange[0]+Erange[1]*(1-enden)
        Ens, vecs = sp.sparse.linalg.eigsh(H,k=Nstates,sigma=target_E)#returns eigenvalues in increasing order
        
        Evecs = np.asarray(vecs.conj().T)#puts the vectors into rows since python's index is row-wise
        
        return(Ens, Evecs)

def filterSzstates(N,S,Jcurr=0,full='False'):
    '''
        Generates full Sz states for total spin Jcurr.
        full = true: Returns all Sz states of the system, not just ones with total spin = Jcurr.
    '''
    #This handy tool generates an increasing binary representation of spin states for N particles, i.e. for N=4:[0000],[0001],[0010]...[1110][1111]
    basiskey = list(itertools.product([0, 1], repeat=N))
    states = []
    for i,basis in enumerate(basiskey):
        
        basis_state = [basis[i] for i in range(len(basis))]
        states.append((basis_state,i))
    
    #Generates all states
    if full == 'True':
        statelist = states
        
    #Filters states with total Sz = Jcurr and generates a list of tuples whos first element is the binary representation of the total spin state, and second element is the state index
    elif full == 'False':
        index = []
        
        for state in states:
            if sum(state[0])-N*S == Jcurr:
                index.append(state[1])
            else:
                index = index
    
        statelist = np.take(states,index,axis=0)
        
    #Generates Sz representation of spin states using tensor products of up and down states, the order is given by the binary representation of the state
    Statevecs = []
    for state in statelist:
        vec = []
        up = [1,0]
        down = [0,1]
        
        for i in state[0]:
            
            if i == 0:
                vec.append(down)
            elif i ==1:
                vec.append(up)
        
        A = vec[0]
        for i in range(len(vec)-1): 
            B = np.kron(A,vec[i+1])
            A = B
         
        Statevecs.append(A)
        
    #Statevecs is originally an array of arrays of arrays, so we squeeze Statevecs which gives us an array of arrays, or a matrix whos rows are the states in the Sz representation
    return(np.squeeze(Statevecs))
    


def filterEnVals(H,Nstates,enden=0.5,full='False'):
    
    #filters eigenvalues only. If you want states with your eigenvalues use filterHstates
    
    if full =='True':
        
        Eigs, vecs = np.linalg.eigh(H.todense())
        Evecs = np.asarray(vecs.conj().T)
        
        return (Eigs,Evecs)
    
    elif full == 'False':
        
        #E = np.linalg.eigvalsh(H.todense())
        def ExmEigs(param):
            E = sp.sparse.linalg.eigsh(H,1,which=param,tol=1e-10,return_eigenvectors=False)#finds a single eigenvalue, in this case it's either the largest or smallest based on params
            return(np.asscalar(E))
        
        params = ['SA','LA']
        Erange= list(map(ExmEigs,params))
        target_E = enden*Erange[0]+Erange[1]*(1-enden)
        
        Ens = sp.sparse.linalg.eigsh(H,k=Nstates,sigma=target_E,return_eigenvectors=False)#returns eigenvalues in increasing order
        
        return(Ens)

