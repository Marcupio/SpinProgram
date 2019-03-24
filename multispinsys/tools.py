# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:05:51 2018

@author: nac2313
"""
import time
import numpy as np
import copy
import itertools
from TensorOps import spinops as so
from TensorOps import spintensor as st
exp = np.exp

#Various support functions used throughout the program. Most involve manipulating the
#eigenvectors/eigenvalues
def ExOp(index1,index2,state):
    
    
    State = copy.deepcopy(state)
    '''
        Python only passes parameters within functions: NEVER copies. 
        I.e state->State where 'State' is NOT a copy of 'state', it is a new name for state and any changes
        to 'State' are reflected in 'state'. You are basically refering to the same object by a different name.
        References to either of the object's names will change the object: two names, one object. 
    
        deepcopy(object) will create a legitamate copy of an object, and anything done to the copy will not be 
        reflected in the original.
    
    Exchange coupling operator between two spin-1/2 particles; gives coupling coefficients and coupled states.
    
        Operator is of the form ExOp = Sz1*Sz2 + 0.5(S1p*S2m + S1m*S2p)
        
        where  S1p(S1m) is the raising(lowering) operator for spin 1 <-particle index.
        
    '''
    #spinflip coefficient
    if State[index1] == State[index2]:
        
        coeff = 0.25 
        State = State
        
        return([(coeff,state)])
        
    if State[index1] == 0 and State[index2] == 1:
        
        
        coeff1 = -0.25
        coeff2 = 0.5
        
        State[index1] = 1
        State[index2] = 0
        
        return([(coeff1,state),(coeff2,State)])
    
    if State[index1] == 1 and State[index2] == 0:
        
        coeff1 = -0.25
        coeff2 = 0.5
        State[index1] = 0
        State[index2] = 1
        
        return([(coeff1,state),(coeff2,State)])
    

def Intpairs(N,nlegs,modex='open',modey='open'):
    
    '''
        Forms a list of tuples which give information on which sites interact, and in
        which direction, for nearest neighbor coupling and are of the form:
            (exchange direction, site index 1, site index 2)
        
    '''
    pairs = []#list of tuples of the form (exchange direction, site index 1, site index 2)
    Nint = int(N/nlegs)
    
    for k in range(0,N):
        
    #handles x direction
        if (k+1)%Nint == 0 and k > 0:
        
            if modex == 'periodic':
            
                pair = (1,int(k),int(k-Nint+1))#gives index pair for site and nearest neighbor along x-direction
                pairs.append(pair)
        
            elif modex == 'open':
            
                pairs = pairs #i.e. nothing happens

        else:
            
            pair = (1,int(k),int(k+1))
            pairs.append(pair)
        
    #handles y direction
        if N-k <= Nint: #covers the last row interactions
        
            if modey == 'periodic':
            
                pair = (2,int(k),int(k-Nint*(nlegs-1)))#gives index pair for site and nearest neighbor along y-direction
                pairs.append(pair)
        
            elif modey == 'open':
        
                pairs = pairs #again nothing happens
            
        else:
        
            pair = (2,int(k),int(k+Nint))
            pairs.append(pair)
           
    return(pairs)



def Statelist(N,S,Jcurr=0,full='False'):
    '''
        Generates a list of all total Sz = Jcurr multi-states for spin 1/2 systems
        e.g. Jcurr = 0 then generates -> [1,0,1,0],[1,1,0,0]...
    
    '''
    basiskey = list(itertools.product([0, 1], repeat=N))
    
    def basis(basiskeyelement):
        
        basis_state = list(basiskeyelement[1])
        return(basis_state,basiskeyelement[0])
    
    states = list(map(basis,enumerate(basiskey)))
    
    '''
    states = []
    for i,basis in enumerate(basiskey):
        
        basis_state = [basis[i] for i in range(len(basis))]
        states.append((basis_state,i))
    '''
    #Generates all states
    if full == 'True':
        statelist = [state[0] for state in states]
        
    #Filters states with total Sz = Jcurr and generates a list of tuples whos first element is the binary representation of the total spin state, and second element is the state index
    elif full == 'False':
        index = []
        
        for state in states:
            if sum(state[0])-N*S == Jcurr:
                index.append(state[1])
            else:
                index = index
    
        slist = np.take(states,index,axis=0)
        statelist = [state[0] for state in slist]
    
    return(list(reversed(statelist)))


def StatelistAlt(inistate):
    
    def next_permutation(arr):
        # Find non-increasing suffix
        i = len(arr) - 1
        while i > 0 and arr[i - 1] >= arr[i]:
            i -= 1
        if i <= 0:
            return False
        
        # Find successor to pivot
        j = len(arr) - 1
        while arr[j] <= arr[i - 1]:
            j -= 1
        arr[i - 1], arr[j] = arr[j], arr[i - 1]
        
        # Reverse suffix
        arr[i : ] = arr[len(arr) - 1 : i - 1 : -1]
        return True
    
    statelist=[]
    state = next_permutation(inistate)
    if state == False:
        statelist=statelist
    
    else:
        while state == True:
            
            statelist.append(copy.deepcopy(inistate))
            state = next_permutation(inistate)
            
    return(reversed(statelist))

def Statedict(listofstates):
    '''
        Generates a dictionary given a list of states(arrays) who keys are the elements of the
        state(array) as a string e.g. [1,0,1,0]->'1010', and values are the index of the state
        in the list of states e.g. [[1,0,1,0],[1,0,1,1]] -> [1,0,1,1] -> key=1011 : value=1
    
    '''
    def key(state):
        return((''.join(map(str,state[1])),state[0]))
    
    #does the equivalent of the commented out code
    statedict = dict(map(key, enumerate(listofstates)))
    
    '''
    for i,state in enumerate(listofstates):
        
        key = ''.join(map(str,state))
        A = (key,i)
        OrderedStates.append(A)
        
        '''
    
    return(statedict)

def expandState(binarystate):
    
    vec = []
    up = [1,0]
    down = [0,1]
        
    for spin in binarystate:
            
        if spin == 0:
            vec.append(down)
        elif spin ==1:
            vec.append(up)
    exstate = vec[0]
    for i in range(len(vec)-1):
        D = np.kron(exstate,vec[i+1])
        exstate = D
    
    return(exstate)

def exval(state1,state2,Op):
    #Performs the operation of bra*operator*ket
    
    A = state1@Op@state2.T
    #A = np.asscalar(a)
    
    return(A)

def timeEv(t,Eneig1,Eneig2,Coeff):
    
    #multiplies each static coefficient by the appropriate time dependent one
    
    A = exp(-1j*t*(Eneig2-Eneig1))*Coeff
    
    return(A)


def Expand(Statevec,S,N,Jcurr):
    
    #expands a state from a subspace with total Sz=Jcurr to full hilbert space of system for spin-1/2 particles
    
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

def initialState(N,S,Jcurr):
    ###generates lowest valued binary state determined by Jcurr i.e. N=6 Jcurr=0 -> [000111]
    state = np.zeros(N,dtype = int)
    i=1
    if sum(state)-N*S == Jcurr:
        state = state
    
    else:
        while sum(state)-N*S < Jcurr:
            
            state[-i] = 1
            i = i +1
    
    return(state)

#test code used for determining how Hamiltonian acts on a ket
'''
def H_on_biket(biket,N,S,nlegs,Js=[1,1],gamma=[0,0],phi=[0,0],full='False'):
    
    def rowfunc(N,nleg,k):
    
        Nint = N/nleg
        A1 = (nleg*k/N - nleg/N*(k%Nint) + 1)-1
        return(int(A1))

    state = biket
    Slist = to.Statelist(N,S,full=full)
    Sdict = to.Statedict(Slist)
    Ipairs = to.Intpairs(N,nlegs)
    Nint = int(N/nlegs)
    
    Vecprime = np.zeros(len(Slist))
    for site in range(0,N):
        
        siteindex = int(site)
        Siteint = []
    #Pulls interaction pairs for the ith site e.g. site=1: S1*S2, S1*S3,...,S1*Sn
        for pair in Ipairs:
            if pair[1] == siteindex:
                Siteint.append(pair)   
            else:
                Siteint = Siteint
                
        for exchange in Siteint:
                    
                #The interaction constants are determined by direction, which is referenced in Intpairs().
                    if exchange[0] == 1:
                        J = Js[0]
                        
                        
                    elif exchange[0] == 2:
                        J = Js[1]
                        
                    
                    m = exchange[1]
                    n = exchange[2]
                    
                    interact = to.ExOp(m,n,state)#returns [(coeff1,state),(coeff2,copuledstate)]
                    
                    for k in interact:
                         
                        j = Sdict[''.join(map(str,k[1]))]
                          
                        Vecprime[j] += J*k[0]
                        
        gamma1 = gamma[0]
        gamma2 = gamma[1]
        phi1 = phi[0]
        phi2 = phi[1]
        disorder1 = []
        disorder2 = []
        
        rowindex = rowfunc(N,nlegs,siteindex)
        colindex = siteindex%Nint
        
        sitephase1 = colindex + 1
        sitephase2 = rowindex + 1
        g1 = gamma1*np.cos(2*np.pi*c*sitephase1 + phi1)#gives disorder only along x-direction(or direction 1)
        g2 = gamma2*np.cos(2*np.pi*c*sitephase2 + phi2)
        for state in Slist:
            
            if state[siteindex] == 0:
                ms = -0.5
                
            elif state[siteindex]== 1:
                ms = 0.5
            
            disorder1.append(ms*g1)
            disorder2.append(ms*g2)
        
        VPrime = np.add(Vecprime,np.add(disorder1,disorder2))
    
    return(VPrime)
'''