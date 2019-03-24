# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:24:16 2019

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
from matplotlib import pyplot as plt

#calculates AGR from saved data

N = 9
o = 'open'
p = 'periodic'
modex=o
modey=o
gamamt1=20
gammax1 = 30
c=np.sqrt(2)
uniform1='False'
FullEigFunc = 'False'

gammas = np.linspace(0.5,gammax1,gamamt1)

Nstates=10
J1=1
J2=1
phiamt=10
Nlegs = 3
Na=4
Nb=5
S = 0.5




#Eigfuncs[gammaindex],[gammavalue=0 vecsvals=1],[phi],[vals=0 vecs=1],[element for vals/vec array]. 
#This is how the data is structured, the vecs are saved as rows, not columns since python is row major or whatever it's called.

str1 = pl.PurePath(modex+'x_'+modey+'y')#sorts by boundary conditions first
c2 = str(round(c,3))#rounds the value of c out to 3 decimal places and converts it to string to be used in file destination
path = pl.PurePath(os.getcwd()).parent/'Data'/str1/'Eigenfunctions' ###path and filepaths are objects which allow the program to locate data
filename = 'EigFuncs__%dx%d_%dphis_J1_%d__J2_%d__%dgammas_%dgammax__%dNstates__%sx_%sy_disorder_onX_onY__c%s__uniform_%s' % (N/Nlegs,Nlegs,phiamt,J1,J2,gamamt1,gammax1,Nstates,modex,modey,c2,uniform1)
filepath = path/filename
 
start= time.time()
#data[gammaindex],[gammavalue=0 vecsvals=1],[phi],[vals=0 vecs=1],[element for vals/vec array]. 
#This is how the data is structured, the vecs are saved as rows, not columns since python is row major or whatever it's called.   
data = np.load(str(filepath)+'.npz')
end = time.time()
print(end-start,'Time to load data')
#need to extract states from data first
vals = data['Evals']

def getAGRgamma(datum):
    
    def getAGR(sortedEn):
                
        Adgphi = []#find all adgr(r) for a given phi, but each phi corresponds to a unique Hamiltonian i.e. H(phi)
        for n in range(1,len(sortedEn)-1):
                
           Enb = sortedEn[n+1]
           Ena = sortedEn[n-1]
           En = sortedEn[n]
           AGn = Enb - En
           AGa = En - Ena
                    
           A = min(AGn,AGa)/max(AGn,AGa)
           Adgphi.append(A)
                    
        return(np.mean(Adgphi))

    sortedEns = [sorted(vals) for vals in datum]#sorts the Eigenvalues for each phi iteration by increasing value.
    
    AGRphis = [getAGR(En) for En in sortedEns]#gets average AGRs for each phi i.e. <r>phi
    #return(the average value of AGRphis i.e. [<r>phi])     
    return(np.mean(AGRphis))

AGR = [getAGRgamma(v) for v in vals]
plt.xlabel('h')
plt.ylabel('r')
plt.plot(gammas,[r for r in AGR],'-o')

    

