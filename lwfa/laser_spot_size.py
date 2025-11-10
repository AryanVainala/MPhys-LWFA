#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 02:39:21 2025

@author: aryan
"""
from scipy.constants import c
from openpmd_viewer.addons import LpaDiagnostics as lpa #import lpa diagnostics
import matplotlib.pyplot as plt #import plotting package

data = lpa('./diags/hdf5/') #Load data as a variable

z_prop = data.t*c * 1e5 #Calculate z position of diagnostic  (micrometerss)
waist = data.iterate(data.get_laser_waist, pol='x',method='rms') #Gets a list of waist values
plt.plot(z_prop, waist)