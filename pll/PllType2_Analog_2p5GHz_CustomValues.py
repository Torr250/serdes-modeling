# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:53:10 2025

@author: adant
"""

#import useful packages
import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp
from si_prefix import si_format


# Insert custom values for the PLL
Fref = 125e6  # Reference Frequency
N = 20  # Multiplication Factor
Rz = 8e3  # Zero Resistor Value
Icp = 90e-6  # Charge Pump current
tau = 60e-12  # Charge Pump Pulse
KVCO_HzV = 1.8e9  # VCO Gain in Hz/V
Cunit = 1e-12  # Unitary Capacitor for Capacitor ratio
C1 = 24 * Cunit
C2 = 0.5 * Cunit

# PLL calculations
f0 = Fref * N  # Output Frequency
KVCOrad = (2 * np.pi) * KVCO_HzV  # VCO gain in rad/s
KVCO = KVCOrad  # For future calculations
KPD = Icp / (2 * np.pi)  # Charge pump Gain

# Transfer function
#               n0 + n1s   
# H(s) = ------------------------
#        d0 + d1s + d2s^2 + d3s^3

#               n1s + n0   
# H(s) = ------------------------
#        d3s^3 + d2s^2 + d1s + d0
# n0 = KPD * KVCO
# n1 = C1 * KPD * KVCO * Rz
# d0 = 0
# d1 = 0
# d2 = (C1 + C2) * N
# d3 = C1 * C2 * N * Rz

import types
LG = types.SimpleNamespace()
CL = types.SimpleNamespace()

# Open Loop
LG.n0 = KPD * KVCO
LG.n1 = C1 * KPD * KVCO * Rz
LG.d0 = 0
LG.d1 = 0
LG.d2 = (C1 + C2) * N
LG.d3 = C1 * C2 * N * Rz
LG.num = [LG.n1, LG.n0]
LG.den = [LG.d3, LG.d2, LG.d1, LG.d0]

# Closed Loop
CL.n0 = KPD * KVCO * N
CL.n1 = C1 * KPD * KVCO * N * Rz
CL.d0 = KPD * KVCO
CL.d1 = C1 * KPD * KVCO * Rz
CL.d2 = C1 * N + C2 * N
CL.d3 = C1 * C2 * N * Rz
CL.num = [CL.n1, CL.n0]
CL.den = [CL.d3, CL.d2, CL.d1, CL.d0]


# Frequency Response
f = np.logspace(3, 9, num=1000)  # Frequency range from 1kHz to 1GHz
w = 2 * np.pi * f  # Angular frequency

# Open Loop Frequency Response
w, H_LG = sp.signal.freqs(LG.num, LG.den, w)
# Closed Loop Frequency Response
w, H_CL = sp.signal.freqs(CL.num, CL.den, w)    
# Plot Bode Plots
plt.figure(dpi=600)
plt.semilogx(1e-6 * f, 20 * np.log10(abs(H_LG)), label='Open Loop')
plt.semilogx(1e-6 * f, 20 * np.log10(abs(H_CL)), label='Closed Loop')
plt.title("PLL Frequency Response")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.legend()
plt.show()

