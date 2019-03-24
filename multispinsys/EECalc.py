# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:57:00 2019

@author: nac2313
"""



import numpy as np
import time
import multiprocessing as mp
import random
from multiprocessing import Pool
import VNEntropy as VN
import tools as t
import os
import pathlib as pl

#calculates EE from saved data
N = 9
o = 'open'
p = 'periodic'
modex=o
modey=o
gamamt1=20
gammax1 = 30
c=np.sqrt(2)
uniform1='False'

gammas = np.linspace(0.5,gammax1,gamamt1)

Nstates=10
J1=1
J2=1
phiamt=10
Nlegs = 3
Na=4#have Na be smaller system if applicable: it's faster
Nb=5
S = 0.5


def bipartDM(State,NA,NB):
    """
        Generates the biparte reduced density matrix for system A
        
        'State': Eigenstate which generates the full density matrix to be partially traced over; needs to be in column form
        'basisvecs': Matrix whos rows are the basis states used to generate the coefficients 
        'NA,NB': Numbers of states in system A and B respectively
    
    """
    
    'The coefficients are simply the basisvecs dotted with the Eigenstate, since each basis vector has a tensor structure of s> = a> x b>'
    'The way we reshape is dependent on how our system A and B are defined. In this case the biparte rhoA matrix can be found using this M matrix'
    
    M = np.reshape(State,(2**NA,2**NB))
    
    'The biparte reduced density matrix for system A in general can be found using M@M.T'
    rhoA = M@M.T
    
    return(rhoA)


str1 = pl.PurePath(modex+'x_'+modey+'y')#sorts by boundary conditions first
c2 = str(round(c,3))#rounds the value of c out to 3 decimal places and converts it to string to be used in file destination
path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Eigenfunctions' ###path and filepaths are objects which allow the program to locate data
filename = 'Eigfuncs__%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%dgammax__%dNstates__%sx_%sy_disorder_onX_onY__c%s__uniform_%s' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt1,gammax1,Nstates,modex,modey,c2,uniform1)
filepath = path/filename
 
start= time.time()
data = np.load(str(filepath)+'.npz')
end = time.time()
print(end-start,'Time to load data')

#need to extract states from data first
vecs = data['Evecs']
#each element in the outside array pertains to a single gamma value. each element in the inner array pertains to a given phi value

def getEEgamma(stateset):
    
    def getEEphi(states):
        #generates <EE>phi
        Exstates = [t.Expand(vec,S,N,Jcurr=0) for vec in states]#expands each state to the Hilbert Space of the full Hamiltonian
        rhoAs = [bipartDM(vec1,Na,Nb) for vec1 in Exstates]#generates a reduced density matrix for each state
        eigsA = [np.linalg.eigvals(rho) for rho in rhoAs]#generates eigenvalues for each reduced density matrix
        EE = np.mean([VN.VNEnt(eigs) for eigs in eigsA])#gives VNEntropy for each state, which is then averaged over to give the EE for a value of phi,i.e. <EE>phi
    
        return(EE)
    
    EEs = [getEEphi(vecs2) for vecs2 in stateset]
    #gives average over phis, i.e. [<EE>phi]
    return(np.mean(EEs))


#if __name__=='__main__':
#p = Pool(2)

start=time.time()
start1 = time.time()
SA = list(map(getEEgamma,vecs))#[getEEgamma(v) for v in vecs]
end1 = time.time()

print(end1-start1,'Time for EE calculations')

    #saves data for Entropy
path1 = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Entropy' ###path and filepaths are objects which allow  
filenameEE = 'EE__%dx%d_%dphis_%dNstates_J1_%d__J2_%d__%dgammas_%dgammax__Na_%d__Nb_%d__%sx_%sy_disorder_onX_onY__c%s__uniform_%s' % (N/Nlegs,Nlegs,phiamt,Nstates,J1,J2,gamamt1,gammax1,Na,Nb,modex,modey,c2,uniform1)
filepathEE = path1/filenameEE
np.save(str(filepathEE),SA)

