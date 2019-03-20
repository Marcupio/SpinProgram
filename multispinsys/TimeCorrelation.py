# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:10:41 2018

@author: crius
"""
import Hamiltonians as H
import numpy as np
import tools as t
import spintensor as st
import spinops as so
import time
import Expand as ex
from matplotlib import pyplot as plt


exp = np.exp
N = 8
nlegs = 4   
S = 0.5
c = np.sqrt(2)
Jcurr=0
J1=1
J2=1

Htot = H.nlegHeisenberg.blockH(N,S,nlegs,c,Js = [1,1],gamma =[0.0, 4.0],full='True')
Hfull = sum(Htot).todense()

eigs, vecs = np.linalg.eigh(Hfull)
Eigvecs = np.asarray(vecs.T)
print(list(eigs))
Envecs = []
for vc in Eigvecs:
    A = ex.Expand(vc,S,N,Jcurr)
    Envecs.append(A)
              
statelist = t.Statelist(N,S)

Statevecs = []
for state in statelist:
    vec = []
    up = [1,0]
    down = [0,1]
        
    for i in state:
            
        if i == 0:
            vec.append(down)
        elif i ==1:
            vec.append(up)
    Basestate = ''.join(map(str,state))
    B = [Basestate,vec[0]]#initializes state so that tensor product can be preformed
    for i in range(len(vec)-1):
        D = np.kron(B[1],vec[i+1])
        B[1] = D
         
    Statevecs.append((B[0],B[1]))
    
####
Basestates = dict(Statevecs)
#print(Basestates)
BS = Basestates['11001100']
s = so.sz(S)
print(t.exval(BS,BS,Hfull))
#####

k=1
Op = so.SziOp(N,S,k,full='True')

def getkey(m,n):
    key = str((2**m)*(2*n + 1) -1)
    return(key)     

Coeff = []
for i,vec1 in enumerate(Eigvecs):
    for j,vec2 in enumerate(Eigvecs):
        
        co = (getkey(i,j),vec2@BS*t.exval(BS,vec1,Op))
        Coeff.append(co)
        
Coeffs = dict(Coeff)

####

Coeff1 = []
for i1,vec1 in enumerate(Eigvecs) :
    for j1,vec2 in enumerate(Eigvecs):
        
        c1 = Coeffs[getkey(i1,j1)]
        co1 = t.exval(vec1,vec2,Op)*c1
        E1 = (getkey(i1,j1),co1)
        Coeff1.append(E1)
        
        
Coeffs1 = dict(Coeff1)
#print(Coeffs)
######
times = np.linspace(0.0,1,10)
#times = [0, 1/9, 2/9,3/9,4/9,5/9,6/9,7/9,8/9,1]
#print(times)
timeEx = []

for tim in times:
    #print(tim)
    start = time.time()
   
    ex = []
    for i,eig1 in enumerate(eigs):
        for j,eig2 in enumerate(eigs):
            
            C = Coeffs1[getkey(i,j)]
            
            #F = t.timeEv(tim,eig1,eig2,C)
            F = C*np.cos(tim*(eig2-eig1))#
            ex.append(F)
      
    timeEx.append(sum(ex))
    
    stop = time.time()
    
    print(stop-start)

print(timeEx)
plt.plot(times,timeEx)