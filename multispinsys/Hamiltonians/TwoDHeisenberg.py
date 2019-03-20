# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:51:34 2018

@author: nac2313
"""

import numpy as np
import scipy as sp
import spinops as so
import spintensor as st
import time
#create heisenberg hamilitonian with periodic/non-periodic boundary conditions
#and nearest neighbor and next nearest neighbor interactions

def H(N,Nlegs,S,J=[1,1],gamma=[0,0],U=0,delx=[1,1],dely=[1,1],delz=[1,1],beta=[1,1],phi=[1,1],mode=['open','open']):
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
    phi1 = phi[0]
    phi2 = phi[1]
    gamma1 = gamma[0]
    gamma2 = gamma[1]
    J1 = J[0]
    J2 = J[1]
    delx1 = delx[0]
    delx2 = delx[1]
    dely1 = dely[0]
    dely2 = dely[1]
    delz1 = delz[0]
    delz2 = delz[1]
    beta1 = beta[0]
    beta2 = beta[1]
    modex = mode[0]
    modey = mode[1]
#generates Hamilitonians of the form Si*Si+1 where Si is a full operator
#e.g. Si = Six + Siy + Siz
    Nint = int(N/Nlegs)
    for k in range(1,N+1):
        
        Sxp = st.spintensor(k, N, sx,S)
        Syp = st.spintensor(k, N, sy,S)
        Szp = st.spintensor(k, N, sz,S)
        
        if k%Nint==0: #handles direction x
           
            if modex == 'periodic':
           
                couple1x = sp.sparse.kron(sp.sparse.identity((2*S+1)**(k-Nlegs-1)), sp.sparse.kron(sx,sp.sparse.identity((2*S+1)**(N-k+Nlegs+1))))
                couple1y = sp.sparse.kron(sp.sparse.identity((2*S+1)**(k-Nlegs-1)), sp.sparse.kron(sy,sp.sparse.identity((2*S+1)**(N-k+Nlegs+1))))
                couple1z = sp.sparse.kron(sp.sparse.identity((2*S+1)**(k-Nlegs-1)), sp.sparse.kron(sz,sp.sparse.identity((2*S+1)**(N-k+Nlegs+1))))
                
           
                Sxp1 = Sxp@couple1x
                Syp1 = Syp@couple1y
                Szp1 = Szp@couple1z
           
            elif modex == 'open':
                
                Sxp1 = sp.sparse.csr_matrix((dim,dim))
                Syp1 = sp.sparse.csr_matrix((dim,dim))
                Szp1 = sp.sparse.csr_matrix((dim,dim))
                
        else:
            
            couple1x = st.spintensor(k+1, N, sx, S)
            couple1y = st.spintensor(k+1, N, sy, S)
            couple1z = st.spintensor(k+1, N, sz, S)
            
            Sxp1 = Sxp@couple1x
            Syp1 = Syp@couple1y
            Szp1 = Szp@couple1z
                
        if N-k < Nint: #handles direction y
            
            if modey == 'periodic':
                
                couple2x = sp.sparse.kron(sp.sparse.identity((2*S+1)**(k-Nint*(Nlegs-1)-1), sp.sparse.kron(sx,sp.sparse.identity((2*S+1)**(N-k+Nint*(Nlegs-1))))))
                couple2y = sp.sparse.kron(sp.sparse.identity((2*S+1)**(k-Nint*(Nlegs-1)-1), sp.sparse.kron(sy,sp.sparse.identity((2*S+1)**(N-k+Nint*(Nlegs-1))))))
                couple2z = sp.sparse.kron(sp.sparse.identity((2*S+1)**(k-Nint*(Nlegs-1)-1), sp.sparse.kron(sz,sp.sparse.identity((2*S+1)**(N-k+Nint*(Nlegs-1))))))
                
                Sxp2 = Sxp@couple2x
                Syp2 = Syp@couple2y
                Szp2 = Szp@couple2z
            
            elif modey == 'open':
                
                Sxp2= sp.sparse.csr_matrix((dim,dim))
                Syp2= sp.sparse.csr_matrix((dim,dim))
                Szp2= sp.sparse.csr_matrix((dim,dim))
    #future work on possible next-nearest neighbor interactions
    #if k==N-1:
        
    #    couple1x = st.spintensor(k+1, N, sx, S)
    #    couple1y = st.spintensor(k+1, N, sy, S)
    #    couple1z = st.spintensor(k+1, N, sx, S)
        
        else:
            
            #couple1x = st.spintensor(k+1, N, sx, S)
            #couple1y = st.spintensor(k+1, N, sy, S)
            #couple1z = st.spintensor(k+1, N, sz, S)
            
            #interactions along other direction
            couple2x = st.spintensor(k+Nint,N,sx,S)
            couple2y = st.spintensor(k+Nint,N,sy,S)
            couple2z = st.spintensor(k+Nint,N,sz,S)
            
        
    #nearest neighbor interaction along both directions 1 and 2
            
            
            #Sxp1 = Sxp@couple1x
            #Syp1 = Syp@couple1y
            #Szp1 = Szp@couple1z
            
            Sxp2 = Sxp@couple2x
            Syp2 = Syp@couple2y
            Szp2 = Szp@couple2z
        
    #disorder interaction
        
        Sk1 = Szp*np.cos(2*np.pi*beta1*k + phi1)
        Sk2 = Szp*np.cos(2*np.pi*beta2*k + phi2)
        
    #same site interaction
        Sk = Szp
        
    #Generate hamiltonian for ith site    
        def interaction(J,xint,yint,zint,siteint,A,B,C,G):
            return J*(A*xint + B*yint + C*zint) + G*siteint
        
        H1 = interaction(J1,Sxp1.real,Syp1.real,Szp1.real,Sk1,delx1,dely1,delz1,gamma1)
        
        #H1 = J1*(delx1*Sxp1.real + dely1*Syp1.real + delz1*Szp1.real) + gamma1*Sk1
        H2 = J2*(delx2*Sxp2.real + dely2*Syp2.real + delz2*Szp2.real) + gamma2*Sk2
        H.append(U*Sk + H1 + H2)
        
        #H1 = U*Sk + J1*(delx*Sxp1.real + dely*Syp1.real + delz*Szp1.real) + gamma*Sk1
        #print(H1)
    
    #H1 = U*Sk + J1*(delx*Sxp1.real + dely*Syp1.real + delz*Szp1.real) + gamma*Sk1

    return(H)
