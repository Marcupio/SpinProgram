# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:53:04 2017
@author: crius
"""

import numpy as np
import scipy as sp
from TensorOps import spinops as so
import time
#create heisenberg hamilitonian with periodic/non-periodic boundary conditions
#and nearest neighbor and next nearest neighbor interactions

def H(N,S,J1=1,gamma=0,U=0,delx=1,dely=1,delz=1,beta=1,phi=0,mode='open'):
    """
    Args: 'N' Number of sites
          'S' maximum spin for a single particle
          'J1' nearest neigbor interaction strength
          'U' same site interaction engergy
          'delx,dely,delz' anisotropy along the X,Y,Z directions
          
          The quasi field is given by the term: cos(2*pi*beta*k + phi)
          'gamma' quasi field interaction strength
          'beta' adjust frequency of quasi field
          'phi' adjust phase of quasi field
    
          'mode' sets boundary condition as 'open' or 'periodic'
    """    
    
    sx = so.sx(S)
    sy = so.sy(S)
    sz = so.sz(S)
    mstates = int(2*S+1)
    dim = mstates**N
    
    H = []

#generates Hamilitonians of the form Si*Si+1 where Si is a full operator
#e.g. Si = Six + Siy + Siz
    
    for k in range(1,N+1):
        
        Sxp = st.spintensor(k, N, sx,S)
        Syp = st.spintensor(k, N, sy,S)
        Szp = st.spintensor(k, N, sz,S)
        
        if k==N:
            
            if mode == 'periodic':
           
                couple1x = sp.sparse.kron(sx, sp.sparse.identity(mstates**(N-1)))
                couple1y = sp.sparse.kron(sy, sp.sparse.identity(mstates**(N-1)))
                couple1z = sp.sparse.kron(sz, sp.sparse.identity(mstates**(N-1)))
           
                Sxp1 = Sxp@couple1x
                Syp1 = Syp@couple1y
                Szp1 = Szp@couple1z
           
            elif mode == 'open':
                
                Sxp1 = sp.sparse.csr_matrix((dim,dim))
                Syp1 = sp.sparse.csr_matrix((dim,dim))
                Szp1 = sp.sparse.csr_matrix((dim,dim))
                
                
    #future work on possible next-nearest neighbor interactions
    #if k==N-1:
        
    #    couple1x = st.spintensor(k+1, N, sx, S)
    #    couple1y = st.spintensor(k+1, N, sy, S)
    #    couple1z = st.spintensor(k+1, N, sx, S)
        
        else:
        
            couple1x = st.spintensor(k+1, N, sx, S)
            couple1y = st.spintensor(k+1, N, sy, S)
            couple1z = st.spintensor(k+1, N, sz, S)
        
    #nearest neighbor interaction
            Sxp1 = Sxp@couple1x
            Syp1 = Syp@couple1y
            Szp1 = Szp@couple1z
        
    #disorder interaction
        Sk1 = Szp*np.cos(2*np.pi*beta*k + phi)
        
    #same site interaction
        Sk = Szp
        
        H.append(U*Sk + J1*(delx*Sxp1.real + dely*Syp1.real + delz*Szp1.real) + gamma*Sk1)
        
        #H1 = U*Sk + J1*(delx*Sxp1.real + dely*Syp1.real + delz*Szp1.real) + gamma*Sk1
        #print(H1)
    
    #H1 = U*Sk + J1*(delx*Sxp1.real + dely*Syp1.real + delz*Szp1.real) + gamma*Sk1

    return(H)
N = 4
S = 0.5
Ham = sum(H(N,S))
fullHam = Ham.todense()
    
    
    