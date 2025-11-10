#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:37:04 2025

@author: aryan
"""

import matplotlib
# matplotlib.use('Qt5Agg')  # Try Qt backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from openpmd_viewer import OpenPMDTimeSeries

# Load the data
ts_2d = OpenPMDTimeSeries('./diags/hdf5/')

ts_2d.slider()

Ex, info_Ex = ts_2d.get_field( t=4000,  field='E', coord='x', plot=True )
