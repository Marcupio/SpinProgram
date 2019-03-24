# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:30:05 2018

@author: crius
"""

import numpy as np
import itertools


#this has been incorporated into the tools module
#expands state to the full Hilbert space dimension, this is needed in the EE calculation
def Expand(Statevec,S,N,Jcurr):
    
    
    basiskey = list(itertools.product([0, 1], repeat=N))
    states = []
    index = []
    
    #Generates binary basis representation tuples states = (state,index)
    for i,basis in enumerate(basiskey):
        
        basis_state = [basis[i] for i in range(len(basis))]
        states.append((basis_state,i))
    
    #state[0] gives the ith state in states
    for state in states:
        if sum(state[0])-N*S == Jcurr: #summing up all the ones and zeros in a state and then subtracting the total system spin (N*S) will give Jcurr for that state
            index.append(state[1])#keeps track of which states are Jcurr states
        else:
            index = index
    
    #here the statevec for a give Jcurr is expanded based on the index e.g. [A B C] ->[0 A 0 0 B 0 C]
    A = Statevec
    State = np.zeros(int((2*S+1)**N))
    j=0
    for value in index:
            
            State[int(value)] = A[j]
            j = j+1
            
    #print(State)
    return(np.asmatrix(State))

        