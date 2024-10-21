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


annots = loadmat('B12_Pulse_FFE.mat')
pulse_ffe = annots['pulse_ffe']
pulse_ffe=pulse_ffe[0,:]

pulse_channel = annots['pulse_channel']
pulse_channel=pulse_channel[:,0]

pulse_channel_ffe = annots['pulse_channel_ffe']
pulse_channel_ffe=pulse_channel_ffe[:,0]

pulse_ui = annots['pulse_ui']
pulse_ui=pulse_ui[0,:]




#annots = loadmat('T20_AC.mat')
#t20_ac_mag = annots['ac_mag']
#t20_ac_mag=t20_ac_mag[0,:]
#t20_ac_freq = annots['ac_freq']
#t20_ac_freq=t20_ac_freq[0,:]



fig, ax = plt.subplots()
ax.plot(pulse_ui, pulse_ffe*1e3, color = 'red', linewidth = 3) 
ax.plot(pulse_ui+4, pulse_channel*1e3, color = 'blue', linewidth = 3) 
ax.plot(pulse_ui+4, pulse_channel_ffe*1e3, color = 'green', linewidth = 3) 

#ax.plot(t, c, color = 'red', linewidth = 3)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '.0f' formatting but don't label
# minor ticks.  The string is used directly, the `StrMethodFormatter` is
# created automatically.
ax.grid(True,linestyle='--',which='both')
ax.minorticks_on()

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter('{x:0.0f}')

ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(100))
ax.yaxis.set_major_formatter('{x:1.0f}')

ax.set_xlim([-2, 8])
ax.set_ylim([-300, 800])


plt.xticks(weight='bold',fontsize=16)
plt.yticks(weight='bold',fontsize=16)

plt.xlabel('Time (UI)',weight='bold',fontsize=16)
plt.ylabel('Voltage (mV)', weight='bold',fontsize=16)


plt.legend(['Input 3 tap FFE','Output w/o Eq', 'Output w/i Eq'],prop = { "size": 16 ,'weight':'bold'}, loc ="upper right")

#ax.text(60, -0.75, 'Test', color='black', weight='bold',fontsize=16, 
        #bbox=dict(facecolor='white', edgecolor='black', pad=5.0)) #round,pad=0.3

#plt.annotate('',xy=(45,-0.45),xytext=(40,-0.75),
             #arrowprops=dict(facecolor='black',width=2,headwidth=10,headlength=10,shrink=0.3))

#ax.add_patch(Rectangle((20, -0.125), 40, 0.25,  edgecolor='black', facecolor="#F8E71C", linewidth=2,linestyle='--'))


#plt.axvline(x = 30, color = 'green', linewidth=2,  linestyle='--')
#plt.axhline(y = -0.95, color = 'green', linewidth=2,  linestyle='--')

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig("plot_pulse_ffe.pdf")

plt.show()