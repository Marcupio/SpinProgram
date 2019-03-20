# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:55:54 2017

@author: nac2313
"""

import numpy as np
import scipy as sp
import tools as to
import itertools


def bipartDM(State,NA,NB):
    """
        Generates the biparte reduced density matrix for system A
        
        'State': Eigenstate which generates the full density matrix to be partially traced over; needs to be in column form
        'basisvecs': Matrix whos rows are the basis states used to generate the coefficients 
        'NA,NB': Numbers of states in system A and B respectively
    
    """
    #coeff = []
    
    'The coefficients are simply the basisvecs dotted with the Eigenstate, since each basis vector has a tensor structure of s> = a> x b>'
    #for row in basisvecs:
        
    #    coeff.append(np.asscalar(np.dot(row,State)))
   
    'The way we reshape is dependent on how our system A and B are defined. In this case the biparte rhoA matrix can be found using this M matrix'
    
    M = np.reshape(State,(2**NA,2**NB))
    
    'The biparte reduced density matrix for system A in general can be found using M@M.T'
    rhoA = M@M.T
    
    return(rhoA)
    
def reducedDM(energystate,NAsites,NBsites,N,S,Jcurr=0,full='False'):
    '''
        energystate: eigen energy state in decreasing binary basis
        NAsites/NBsites: arrays with corresponding site indecies for system A/B
        N: total number of sites
        S: total spin of each site
        
    '''
    
    '''
    Generate coefficients
    '''
    coeffs1 = [(i,coeff) for i,coeff in enumerate(energystate)]
    coeffdict1 = dict(coeffs1)
    
    ###Generate list and dictionary of full binary basis with total spin Jcurr
    Slist = to.Statelist(N,S,Jcurr=Jcurr,full=full)
    Sdict = to.Statedict(Slist)
    
    ####Generate system-basis based off size of system A/B
    Na = len(NAsites)
    Nb = len(NBsites)
    
    basiskeyA = list(itertools.product([0, 1], repeat=Na))
    statesA = []
    for i,basis in enumerate(basiskeyA):
            
        basis_state = [basis[i] for i in range(len(basis))]
        statesA.append(basis_state)
    
    
    basiskeyB = list(itertools.product([0, 1], repeat=Nb))
    SAdict = to.Statedict(statesA)
    statesB = []
    for i,basis in enumerate(basiskeyB):
            
        basis_state = [basis[i] for i in range(len(basis))]
        statesB.append((i,basis_state))
        
    ####Figure out which states in basis B are connected to A via full binary state
    
    Bconnect = []
    
    for stateB in statesB:
        connected = [stateB[0],[]]###array used to form dictionary
        for states in Slist:
                
                B = [states[i2] for i2 in NBsites]#used to compare overall state with stateB
                
                if stateB[1] == B: #if stateB and overall state are compatible
                    A = [states[i1] for i1 in NAsites]###finds connected A state
                    stateindex = Sdict[''.join(map(str,states))]###finds stateindex to look up coefficient
                    stateindexA = SAdict[''.join(map(str,A))]###finds index of connected A state
                    ###stores index of state A, and corresponding coefficient for a given State B
                    connected[1].append((stateindexA,coeffdict1[stateindex]))
                
                else:
                    
                    connected = connected
        
        Bconnect.append(connected)
        
    ###connectD stores all necessary information to find out which states are entangled,
    ###and what the coefficient is
    
    connectD = dict(Bconnect)
    dim = int(len(statesA))#rhoA is based on dimensions of system A
    rho = []
    for i in range(len(statesB)):
        trace = connectD[i]#pulls up specific info for a given B state
        row = []
        col = []
        data = []
        '''
            Essentially all state indicies for the given A states are iterated over,
            the corresponding coefficients are connected to state A given a specific
            state B.
        '''
        for info1 in trace:
            for info2 in trace:
                
                row.append(info1[0])
                col.append(info2[0])
                data.append(info1[1]*info2[1].conjugate())
        
        rhoAB = sp.sparse.coo_matrix((data,(row,col)),shape=(dim,dim))
        rho.append(rhoAB)
    
    rhoA = sum(rho)
    
    return(rhoA)
    
def VNEnt(EigsA):
    """
        Filters eigenvalues to deal with 0*ln(0) terms within the Von Neuman entropy summation
        
        'EigsA': List of eigenvalues of a corresponding density matrix RhoA
        
    """
    
    FilteredEigs = [Eigs  if abs(Eigs) > 1e-8 else 1 for Eigs in EigsA]
    
    SA = -sum(FilteredEigs*np.log(FilteredEigs))
    
    return(SA)
