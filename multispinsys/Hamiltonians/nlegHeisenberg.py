# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:42:44 2018

@author: nac2313
"""

import numpy as np
import random
import FilteredStates as fs
import time
import scipy as sp
import tools as t

#Generates the Hamiltonian for each site in an mxn square lattice of 1/2-spins as a list of on-site Hamiltonians
#To get the system's full Hamiltonian, simply sum(H) where H is what is returned.

def rowfunc(N,nleg,k):
    #determines row index for site i for a n x nleg lattice
    Nint = N/nleg
    A = (nleg*k/N - nleg/N*(k%Nint) + 1)-1
    return(int(A))

def blockH(N,nlegs,c,S=0.5,TotalSz=0,Js=[1,1],gamma=[0,0],phi=[0,0],modex1='open',modey1='open',full='False', uniform='False'):
    
    #We often only look at state with some total Sz, the default is Sz=0. If we wish to look
    #at the Full, block-diagonalized Hamiltonian simply set full='True'
    #be warned that the Hilbert dimensions for the Full Hamiltonian goes as 2**N where N is the number of spins.
    #Exploring the Full Hamiltonian is incredibly resource intensive for the average computer.
    
    #nlegs is along x-direction i.e. it gives the number or columns or legs of the ladder
    #N/nlegs gives the number of rows
    #Generating the StateAltlist is faster, but accessing it is slower
    Slist = t.Statelist(N,S,Jcurr=TotalSz,full=full)
    Sdict = t.Statedict(Slist)#generates a dictionary where each state has an index, and so we have an ordered basis
    Ipairs = t.Intpairs(N,nlegs,modex=modex1,modey=modey1)#generates interaction pairs as tuples
    
    Nint = int(N/nlegs)
    H = []#our list of site Hamiltonians
    
    
###Here we generate the site Hamiltonian for the ith spin
    
    for site in range(0,N):
        
        siteindex = int(site)
        Siteint = []
        #Pulls interaction pairs for the ith site e.g. site=1: S1*S2, S1*S3,...,S1*Sn
        for pair in Ipairs:
            if pair[1] == siteindex:
                Siteint.append(pair)   
            else:
               Siteint = Siteint

        #prepare information to generate sparse site-Hamiltonian
        rows = []
        cols = []
        data = []
        
        for i,state in enumerate(Slist):
            for exchange in Siteint:
                
            #The interaction constants are determined by direction, which is referenced in Intpairs().
                if exchange[0] == 1:
                    J = Js[0]
                    
                    
                elif exchange[0] == 2:
                    J = Js[1]
                    
                
                m = exchange[1]
                n = exchange[2]
                
                interact = t.ExOp(m,n,state)#returns [(coeff1,state),(coeff2,copuledstate)]
                
                '''
                Where the magic happens. Here we reference the dictionary to see which states couple with 'state',
                and we record the coordinates and coefficients. k is a tuple (coeff,state), k[1] gives the state 
                which is used as a key to find the column index. Together with the index i, we have the coordinate
                for the coefficient given by k[0].
                '''
                
                for k in interact:
                     
                    j = Sdict[''.join(map(str,k[1]))]#here we reference the dictionary to determine j
                        
                    rows.append(i)
                    cols.append(j)
                    data.append(J*k[0])
                    
        
        #generate disorder terms, which only lie on the diagonal, along a direction or combination thereof
        gamma1 = gamma[0]
        gamma2 = gamma[1]
        phi1 = phi[0]
        phi2 = phi[1]
        disorder1 = []
        disorder2 = []
        
        rowindex = rowfunc(N,nlegs,siteindex)#gets row index 
        colindex = siteindex%Nint
        
        sitephase1 = colindex + 1
        sitephase2 = rowindex + 1
        
        ##Here we account for the regime where we only want to look at uniform randomness for our disorder term.
        if uniform == 'True':
            
            g1 = random.uniform(-gamma1,gamma1)
            g2 = 0
            
        elif uniform == 'False':
            
            phi1=phi[0]
            phi2=phi[1]
        
        
        #technically these are just two disorder terms that are added to the hamiltonian. Their dependence on the x(y)-direction is simply due to the model; they can be anything really.
            g1 = gamma1*np.cos(2*np.pi*c*sitephase1 + phi1)#gives disorder along x-direction
            g2 = gamma2*np.cos(2*np.pi*c*sitephase2 + phi2)#gives disorder along y-direction
            
        #this determines if the ith-spin is spin up or down and gives the appropriate spin value to be multiplied to our disorder terms
        for state in Slist:
            
            if state[siteindex] == 0:
                ms = -0.5
                
            elif state[siteindex]== 1:
                ms = 0.5
            
            disorder1.append(ms*g1)
            disorder2.append(ms*g2)
       #generates sparese, disorder matrices for the ith spin
        dim = int(len(Slist))
        disorderH1 = sp.sparse.diags(disorder1)
        disorderH2 = sp.sparse.diags(disorder2)
        
        # each site Hamiltonian is simply the Heisenberg interaction, plus any disorder.
        interactionH = sp.sparse.coo_matrix((data,(rows,cols)),shape=(dim,dim))
        
        siteH = interactionH + disorderH1 + disorderH2
        
        H.append(siteH)
        
    return(H)


    
    