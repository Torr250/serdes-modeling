# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:16:27 2024

@author: adant
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.patches import Rectangle
from scipy.io import loadmat

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


annots = loadmat('B12_Pulse_DFE.mat')

pulse_channel = annots['pulse_channel']
pulse_channel=pulse_channel[:,0]

pulse_channel_dfe = annots['pulse_channel_dfe']
pulse_channel_dfe=pulse_channel_dfe[:,0]

pulse_ui = annots['pulse_ui']
pulse_ui=pulse_ui[0,:]



n_precursors =  3
n_postcursors = 10

n_cursors = n_precursors + n_postcursors + 1
channel_coefficients = np.zeros(n_cursors)


samples_per_symbol = 125
max_idx = np.where(pulse_channel == np.amax(pulse_channel))[0][0]


samples_ui = np.zeros(n_cursors)
samples_channel = np.zeros(n_cursors)
samples_channel_dfe = np.zeros(n_cursors)
samples_channel_sign = np.zeros(n_cursors)

for cursor in range(-n_precursors,n_postcursors+1):
    #print(cursor+n_precursors)
    samples_ui[cursor+n_precursors] = pulse_ui[max_idx + cursor*samples_per_symbol]
    samples_channel[cursor+n_precursors] = pulse_channel[max_idx + cursor*samples_per_symbol]
    samples_channel_dfe[cursor+n_precursors] = pulse_channel_dfe[max_idx + cursor*samples_per_symbol]

samples_channel_sign = np.sign(samples_channel);

#annots = loadmat('T20_AC.mat')
#t20_ac_mag = annots['ac_mag']
#t20_ac_mag=t20_ac_mag[0,:]
#t20_ac_freq = annots['ac_freq']
#t20_ac_freq=t20_ac_freq[0,:]



fig, ax = plt.subplots()
ax.plot(pulse_ui, pulse_channel*1e3, color = 'blue', linewidth = 3) 
#ax.plot(pulse_ui, pulse_channel_dfe*1e3, color = 'red', linewidth = 3) 
ax.plot(samples_ui, samples_channel*1e3, 'o', color = 'blue', linewidth = 1, markersize=10)
#ax.plot(samples_ui, samples_channel_dfe*1e3, 'o', color = 'red', linewidth = 1, markersize=10)
#ax.plot(samples_ui, samples_channel_sign*300, 'o', color = 'blue', linewidth = 1, markersize=10)

plt.legend(['Pulse Response','Samples y[k]'],prop = { "size": 16 ,'weight':'bold'}, loc ="upper right")


#ax.plot(t, c, color = 'red', linewidth = 3)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '.0f' formatting but don't label
# minor ticks.  The string is used directly, the `StrMethodFormatter` is
# created automatically.
ax.grid(True,linestyle='--',which='major',axis = 'y')
ax.minorticks_on()

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter('{x:0.0f}')

ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_formatter('{x:1.0f}')

ax.set_xlim([-n_precursors, n_postcursors])
#ax.set_ylim([-50, 250])


plt.xticks(weight='bold',fontsize=16)
plt.yticks(weight='bold',fontsize=16)

plt.xlabel('Time (UI)',weight='bold',fontsize=16)
plt.ylabel('Voltage (mV)', weight='bold',fontsize=16)


#plt.legend(['No Eq','3 tap DFE'],prop = { "size": 16 ,'weight':'bold'}, loc ="upper right")

#ax.text(60, -0.75, 'Test', color='black', weight='bold',fontsize=16, 
        #bbox=dict(facecolor='white', edgecolor='black', pad=5.0)) #round,pad=0.3

#plt.annotate('',xy=(45,-0.45),xytext=(40,-0.75),
             #arrowprops=dict(facecolor='black',width=2,headwidth=10,headlength=10,shrink=0.3))

#ax.add_patch(Rectangle((20, -0.125), 40, 0.25,  edgecolor='black', facecolor="#F8E71C", linewidth=2,linestyle='--'))


#plt.axvline(x = 30, color = 'green', linewidth=2,  linestyle='--')
#plt.axhline(y = -0.95, color = 'green', linewidth=2,  linestyle='--')

for cursor in range(-n_precursors,n_postcursors):
    plt.axvline(x=cursor+0.5,color = 'grey', linewidth = 0.25, linestyle='--')


plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig("plot_pulse_samples2.pdf")
plt.show()