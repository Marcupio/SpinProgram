# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 16:31:41 2018

@author: nac2313
"""

import numpy as np
import FilteredStates as fs
import time
import scipy as sp
import tools as t
#Exchange operator for Exchange coupling between two sites 1 and 2, for spin-1/2 system

def rowfunc(N,nleg,k):
    
    Nint = N/nleg
    A = (nleg*k/N - nleg/N*(k%Nint) + 1)-1
    return(int(A))

def blockH(N,S,nlegs,c,TotalSz=0,Js=[1,1],gamma=[0,0],phi=[0,0],full='False'):
    
    
    #inistate=t.initialState(N,S,TotalSz)
    #Generating the StateAltlist is faster, but accessing it is slower
    Slist = t.Statelist(N,S,Jcurr=TotalSz,full=full)
    Sdict = t.Statedict(Slist)
    Ipairs = t.Intpairs(N,nlegs)#Intpairs(N,nlegs)
    
    Nint = int(N/nlegs)
    H = []
    
    #print(Slist)
###Here we generate the site Hamiltonian for the ith spin
    
    for site in range(0,N):
        start = time.time()
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
        
        
        
        for exchange in Siteint:
                
            #The interaction constants are determined by direction, which is referenced in Intpairs().
            if exchange[0] == 1:
                J = Js[0]
                    
                    
            elif exchange[0] == 2:
                J = Js[1]
                    
                
            m = exchange[1]#site1 index
            n = exchange[2]#site2 index
            
            ####
            
            def interaction(interinfo):
                
                i = interinfo[0]
                def getrowcol(iter1):
                    j = Sdict[''.join(map(str,iter1[1]))]
                    
                    return(i,j,J*iter1[0])
                
                A = t.ExOp(m,n,interinfo[1])
                
                #list of tuples: info for row and col indicies with corresponding matrix coefficient:(row,col,coeff)
                #for a corresponding state(row)
                row_col_data = map(getrowcol,A)
                
                return(row_col_data)
            
                
            row_col_info = list(map(interaction,enumerate(Slist)))
            for item in row_col_info:
                print(list(item))
            
            
            ####
            '''
            for i,state in enumerate(Slist):    
                interact = t.ExOp(m,n,state)#Performs interaction between site1 and site2: returns [(coeff1,state),(coeff2,copuledstate)]
                
                
                Where the magic happens. Here we reference the dictionary to see which states couple with 'state',
                and we record the coordinates and coefficients. k is a tuple (coeff,state), k[1] gives the state 
                which is used as a key to find the column index. Together with the index i, we have the coordinate
                for the coefficient given by k[0].
                
                
                for k in interact:
                    # 
                    j = Sdict[''.join(map(str,k[1]))]
                        
                    rows.append(i)
                    cols.append(j)
                    data.append(J*k[0])
               '''   
        
        #generate disorder terms, which only lie on the diagonal, along direction 1
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
       
        dim = int(len(Slist))
        disorderH1 = sp.sparse.diags(disorder1)
        disorderH2 = sp.sparse.diags(disorder2)
        
        interactionH = sp.sparse.coo_matrix((data,(rows,cols)),shape=(dim,dim))
        
        siteH = interactionH + disorderH1 + disorderH2
        
        H.append(siteH)
        end = time.time()
        #print(end-start)
    return(H)