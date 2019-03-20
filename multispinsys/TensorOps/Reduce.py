# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:40:20 2018

@author: crius
"""
import numpy as np
import spinops as so
import spintensor as st
import scipy as sp
import time

def Reduce(H,Basis_states):
    
    #HBs = [H[int(i)] for i in HBsites] 
    Hfull = sum(H)#.todense()
    #HB = sum(HBs)#.todense()
    rows = Basis_states  
    
    
###Setup matricies to be multiplied together
    
    #HfullR = np.squeeze(np.asarray(Hfull))
    #HBR = np.squeeze(np.asarray(HB))
    #rowR = [row.tolist() for row in SvecsfilR]
    #rows = [np.squeeze(row).conjugate() for row in rowR]
    cols = rows.conj().T
###Generate sparse Matricies
 
    rowsp = sp.sparse.coo_matrix(rows)
    colsp = sp.sparse.coo_matrix(cols)

### Generate Reduced Hamiltonians
    
    H1 = rowsp@Hfull@colsp
    #H2 = rowsp*HB*colsp#np.linalg.multi_dot([rows,HBR,cols])
    
    return(H1)#,H2.todense())