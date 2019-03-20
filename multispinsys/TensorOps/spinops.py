# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:07:20 2017

@author: crius
"""

import numpy as np
import tools as to
import scipy as sp
#def spinops(S):


#Sx = spinops(S)
#print(Sx)
def raising(S):
    
    SV = np.linspace(S,-S, num=int(2*S+1))#spin values ranging from -S to S in integer steps
    M = np.zeros((np.int(2*S+1)))#initialize column matrix of m's
    splus = np.zeros((int(2*S+1),int(2*S+1)))
    #sminus = np.zeros((int(2*S+1),int(2*S+1)))

    for i in range(len(SV)):
        for j in range(len(SV)):
            M[j] = S - (j+1) + 1
            
            if j==i+1:
                splus[i,j] = np.sqrt((S-M[j])*(S+M[j]+1))
                
            
            #if j== i-1:    
            #    sminus[i,j] = np.sqrt((S+M[j])*(S-M[j]+1))
                    
            else:
                splus = splus
               # sminus = sminus
            
           
    
    return(splus)


def lowering(S):
    
    return raising(S).T

def sx(S):
    
    return 0.5*(raising(S) + lowering(S))

def sy(S):
    
     return -0.5j*(raising(S) - lowering(S))

def sz(S):
    
    return np.diag(np.linspace(S,-S, num=int(2*S+1)))


def SziOp(N,S,index, Jcurr=0, full='False'):
    
    Slist = to.Statelist(N,S,Jcurr=Jcurr,full=full)
    Szidiag = []
    for state in Slist:
        
        if state[index-1] == 1:
            sign = 1
        
        elif state[index-1]== 0:
        
            sign = -1
        
        Szidiag.append(sign*0.5)
    
    Szi = sp.sparse.diags(Szidiag)
    
    return(Szi)

    
    