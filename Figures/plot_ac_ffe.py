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

annots = loadmat('B12_AC_FFE.mat')
channel_ac_mag = annots['channel_ac_mag']
channel_ac_mag=channel_ac_mag[0,:]
channel_ffe_ac_mag = annots['channel_ffe_ac_mag']
channel_ffe_ac_mag=channel_ffe_ac_mag[0,:]
ffe_ac_mag = annots['ffe_ac_mag']
ffe_ac_mag=ffe_ac_mag[:,0]
ac_freq = annots['ac_freq']
ac_freq=ac_freq[0,:]


#annots = loadmat('T20_AC.mat')
#t20_ac_mag = annots['ac_mag']
#t20_ac_mag=t20_ac_mag[0,:]
#t20_ac_freq = annots['ac_freq']
#t20_ac_freq=t20_ac_freq[0,:]



fig, ax = plt.subplots()
ax.plot(ac_freq, channel_ac_mag, color = 'blue', linewidth = 3)
ax.plot(ac_freq, ffe_ac_mag, color = 'red', linewidth = 3)
ax.plot(ac_freq, channel_ffe_ac_mag, color = 'green', linewidth = 3)

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

ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_formatter('{x:1.0f}')

ax.set_xlim([0, 8])
#ax.set_ylim([-1, 1])


plt.xticks(weight='bold',fontsize=16)
plt.yticks(weight='bold',fontsize=16)

plt.xlabel('Frequency (GHz)',weight='bold',fontsize=16)
plt.ylabel('Channel Response (dB)', weight='bold',fontsize=16)


plt.legend(['Channel','TX FFE', 'Channel w\i TX FFE'],prop = { "size": 16 ,'weight':'bold'}, loc ="lower left")

#ax.text(60, -0.75, 'Test', color='black', weight='bold',fontsize=16, 
        #bbox=dict(facecolor='white', edgecolor='black', pad=5.0)) #round,pad=0.3

#plt.annotate('',xy=(45,-0.45),xytext=(40,-0.75),
             #arrowprops=dict(facecolor='black',width=2,headwidth=10,headlength=10,shrink=0.3))

#ax.add_patch(Rectangle((20, -0.125), 40, 0.25,  edgecolor='black', facecolor="#F8E71C", linewidth=2,linestyle='--'))


#plt.axvline(x = 30, color = 'green', linewidth=2,  linestyle='--')
#plt.axhline(y = -0.95, color = 'green', linewidth=2,  linestyle='--')

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('plot_ac_ffe.pdf')  

plt.show()