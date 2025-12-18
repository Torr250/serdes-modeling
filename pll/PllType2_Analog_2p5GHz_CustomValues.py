# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:53:10 2025

@author: adant
"""

#import useful packages
try:
	import serdespy as sdp
except Exception:
	sdp = None
	print('Warning: serdespy not available; continuing without it.')
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp
import control as ct
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

# Open Loop Frequency Response (magnitude and phase)
w, H_LG = sp.signal.freqs(LG.num, LG.den, w)
H_LG_mag_db = 20 * np.log10(np.abs(H_LG))
H_LG_phase_deg = np.unwrap(np.angle(H_LG)) * 180.0 / np.pi

#Gain bandwidth calculation
# Find frequency where magnitude crosses 0 dB
gain_bandwidth = np.nan
for i in range(1, len(H_LG_mag_db)):
    if H_LG_mag_db[i-1] > 0 and H_LG_mag_db[i] <= 0:
        # Linear interpolation to find exact crossover frequency
        f1 = f[i-1]
        f2 = f[i]
        mag1 = H_LG_mag_db[i-1]
        mag2 = H_LG_mag_db[i]
        gain_bandwidth = f1 + (0 - mag1) * (f2 - f1) / (mag2 - mag1)
        break
if np.isfinite(gain_bandwidth):
    print('Gain Bandwidth: ' + si_format(gain_bandwidth) + 'Hz')

#Phase Margin calculation
# Find phase at gain crossover frequency
phase_margin = np.nan
if np.isfinite(gain_bandwidth):
    for i in range(1, len(f)):
        if f[i-1] < gain_bandwidth <= f[i]:
            phase1 = H_LG_phase_deg[i-1]
            phase2 = H_LG_phase_deg[i]
            phase_at_gbw = phase1 + (gain_bandwidth - f[i-1]) * (phase2 - phase1) / (f[i] - f[i-1])
            phase_margin = 180 + phase_at_gbw  # PM = 180 + phase at crossover
            break
if np.isfinite(phase_margin):
    print('Phase Margin: ' + si_format(phase_margin) + ' degrees')

# Closed Loop Frequency Response (magnitude)
w, H_CL = sp.signal.freqs(CL.num, CL.den, w)
H_CL_mag_db = 20 * np.log10(np.abs(H_CL))

# Closed Loop step response (time-domain)
# choose a time vector that spans a few natural periods based on f0
t = np.linspace(0, 1000 / Fref, 1000)
t_out, H_CL_step = sp.signal.step((CL.num, CL.den), T=t)

# Settling time calculation (2% criterion) — using the step response
sys = ct.tf(CL.num, CL.den)
# Compute settling time (2% criterion) directly from the step response arrays
final_value = H_CL_step[-1]
tol = 0.02 * abs(final_value) if np.isfinite(final_value) and final_value != 0 else 0.02
settling_time = np.nan
for idx in range(len(H_CL_step)):
	if np.all(np.abs(H_CL_step[idx:] - final_value) <= tol):
		settling_time = t_out[idx]
		break
if np.isfinite(settling_time):
	print('Settling time (2% criterion): ' + si_format(settling_time) + 's')
else:
	print('Settling time (2% criterion): N/A')

# Plot four independent subplots (2x2)
fig, axs = plt.subplots(2, 2, figsize=(4, 4), dpi=200)

# Top-left: Open Loop magnitude
axs[0, 0].semilogx(1e-6 * f, H_LG_mag_db, color='tab:blue')
axs[0, 0].set_title('Open Loop Magnitude |H_LG| (dB)')
axs[0, 0].set_xlabel('Frequency (MHz)')
axs[0, 0].set_ylabel('Magnitude (dB)')
axs[0, 0].grid(True)

# Top-right: Open Loop phase
axs[0, 1].semilogx(1e-6 * f, H_LG_phase_deg, color='tab:green')
axs[0, 1].set_title('Open Loop Phase ∠H_LG (deg)')
axs[0, 1].set_xlabel('Frequency (MHz)')
axs[0, 1].set_ylabel('Phase (deg)')
axs[0, 1].grid(True)

# Bottom-left: Closed Loop magnitude
axs[1, 0].semilogx(1e-6 * f, H_CL_mag_db, color='tab:orange')
axs[1, 0].set_title('Closed Loop Magnitude |H_CL| (dB)')
axs[1, 0].set_xlabel('Frequency (MHz)')
axs[1, 0].set_ylabel('Magnitude (dB)')
axs[1, 0].grid(True)

# Bottom-right: Closed Loop step response
axs[1, 1].plot(t_out, H_CL_step, color='tab:red')
axs[1, 1].set_title('Closed Loop Step Response')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Normalized amplitude')
axs[1, 1].grid(True)

plt.suptitle('PLL Responses')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
