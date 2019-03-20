# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:34:59 2018

@author: nac2313
"""
import os
import pathlib as pl
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def semiPoisson(agr,beta=1):
    
    r = ((agr+agr**2)**beta)/((1+agr+agr**2)**(1+1.5*beta))
    
    return(r)

AGRs = np.linspace(0,1,100)
Distr = [semiPoisson(rs) for rs in AGRs]

plt.plot(AGRs,Distr)