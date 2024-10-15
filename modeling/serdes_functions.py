#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:49:32 2024

@author: atorralba
"""

import serdespy as sdp
import numpy as np
import scipy as sp
#import serdes_functions as sdf
from si_prefix import si_format
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def channel_coefficients(pulse_response, samples_per_symbol, n_precursors, n_postcursors):
    #TODO: check for not long enough signal
    #make t optional arg
    """measures and plots channel coefficients from pulse response

    Parameters
    ----------
    pulse_response: array
    
    t: array
        time vector corresponding to pulse response

    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    res : int
        resolution of plot
        
    title: str
        title of plot
    
    Returns
    -------
    channel_coefficients : array
        channel coefficents measured from pulse response

    """
    #number of channel coefficients
    n_cursors = n_precursors + n_postcursors + 1
    channel_coefficients = np.zeros(n_cursors)
    
    
    #find peak of pulse response = main cursor sample time
    max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]
    
    for cursor in range(n_cursors):
        
        #index of channel coefficint
        a = cursor - n_precursors
        
        #measure pulse response
        channel_coefficients[cursor] = pulse_response[max_idx+a*samples_per_symbol]
        
    
    return channel_coefficients



def wc_eyeheight(pulse_response, samples_per_symbol, n_precursors, n_postcursors):
    """measures the worst case eye height using the coefficient method

    Parameters
    ----------
    pulse_response: array
    
    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    Returns
    -------
    wc_eyeheight : int
        worst case eye height in volts

    """
    
    ch1_coeff = channel_coefficients(pulse_response, samples_per_symbol, n_precursors, n_postcursors)

    pulse_peak = ch1_coeff[n_precursors]
    WC0 = 0
    WC1 = 0

    for x in range(n_precursors):
        if ch1_coeff[x] < 0 :
            WC1 = ch1_coeff[x] + WC1;
        else:
            WC0 = ch1_coeff[x] + WC0;
            
    for x in range(n_postcursors):
        if ch1_coeff[n_precursors+1+x] < 0 :
            WC1 = ch1_coeff[n_precursors+1+x] + WC1;
        else:
            WC0 = ch1_coeff[n_precursors+1+x] + WC0;
            
    WCEyeH = 2*(pulse_peak+WC1-WC0)

    #print('WC0 (positive): '+si_format(WC0)+', WC1 (negative): '+si_format(WC1)+', Peak, : '+si_format(pulse_peak))
    #print('WC eye height: '+si_format(WCEyeH))
    
    return WCEyeH
    

def wc_eyeheight_coeff(coeff, samples_per_symbol, n_precursors, n_postcursors):
    """measures the worst case eye height using the coefficient method

    Parameters
    ----------
    coeff: array
        Only channel coefficients
    
    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    Returns
    -------
    wc_eyeheight : int
        worst case eye height in volts

    """
    
    pulse_peak = coeff[n_precursors]
    WC0 = 0
    WC1 = 0

    for x in range(n_precursors):
        if coeff[x] < 0 :
            WC1 = coeff[x] + WC1;
        else:
            WC0 = coeff[x] + WC0;
            
    for x in range(n_postcursors):
        if coeff[n_precursors+1+x] < 0 :
            WC1 = coeff[n_precursors+1+x] + WC1;
        else:
            WC0 = coeff[n_precursors+1+x] + WC0;
            
    WCEyeH = 2*(pulse_peak+WC1-WC0)

    #print('WC0 (positive): '+si_format(WC0)+', WC1 (negative): '+si_format(WC1)+', Peak, : '+si_format(pulse_peak))
    #print('WC eye height: '+si_format(WCEyeH))
    
    return WCEyeH


def wc_datapattern(pulse_response, samples_per_symbol, n_precursors, n_postcursors):
    """measures the worst case eye height using the coefficient method

    Parameters
    ----------
    pulse_response: array
    
    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    Returns
    -------
    wc_eyeheight : int
        worst case eye height in volts

    """
    
    ch1_coeff = channel_coefficients(pulse_response, samples_per_symbol, n_precursors, n_postcursors)
    
    pattern_zero = -1*np.sign(ch1_coeff)
    pattern_zero[n_precursors] = -1
    pattern_ones = -1*np.flip(pattern_zero)
    pattern_zo = np.concatenate((pattern_zero, pattern_ones))
    pattern = np.zeros(len(pattern_zo))
    
    for i in range(len(pattern_zo)):
        if pattern_zo[i]>0:
            pattern[i] = 1
        else:
            pattern[i] = 0

    
    return pattern


def simple_eye(signal, window_len, ntraces, tstep, title, res=600, linewidth=0.15):
    """Genterates simple eye diagram

    Parameters
    ----------
    signal: array
        signal to be plotted
    
    window_len: int
        number of time steps in eye diagram x axis
    
    ntraces: int
        number of traces to be plotted
    
    tstep: float
        timestep of time domain signal
    
    title: 
        title of the plot
        
    res: int, optional
        DPI resolution of the figure
        
    linewidth: float, optional
        width of lines in figure
    """
    offset = window_len*0.5
    signal_crop = signal[int(offset):int(((ntraces-1)*window_len+offset))]
    traces = np.split(signal_crop,(ntraces-1))

    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    t_lim = np.floor(1e12*tstep*(window_len-1)/2)+1
    
    #print(t_lim)
    #t_ui = (t*1e12)/t_lim
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots()
    
    for i in range(ntraces-1):
        ax.plot(t*1e12,np.reshape((traces[i][:]),window_len)*1e3, color = 'blue', linewidth = 1)
        
    ax.grid(True,linestyle='--',which='both')
    ax.minorticks_on()
    
    #In picoseconds
    ax.xaxis.set_major_locator(MultipleLocator(t_lim/2))
    ax.xaxis.set_minor_locator(MultipleLocator(t_lim/4))
    ax.xaxis.set_major_formatter('{x:1.0f}')
    
    #In UI
    #ax.xaxis.set_major_locator(MultipleLocator(0.5))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    #ax.xaxis.set_major_formatter('{x:1.1f}')
    
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_minor_locator(MultipleLocator(250))
    ax.yaxis.set_major_formatter('{x:1.0f}')
    
    #ax.set_xlim([-1, 1])
    ax.set_xlim([-t_lim, t_lim])
    ax.set_ylim([-500, 500])
    
    plt.xticks(weight='bold',fontsize=20)
    plt.yticks(weight='bold',fontsize=20)
    
    plt.xlabel('Time (ps)',weight='bold',fontsize=20)
    #plt.xlabel('Time (UI)',weight='bold',fontsize=18)
    plt.ylabel('Amplitude (mV)', weight='bold',fontsize=20)

    #plt.savefig('line_plot.pdf')  
    return True
    
def simple_eye(signal, window_len, ntraces, tstep, title, res=600, linewidth=0.15):
    """Genterates simple eye diagram

    Parameters
    ----------
    signal: array
        signal to be plotted
    
    window_len: int
        number of time steps in eye diagram x axis
    
    ntraces: int
        number of traces to be plotted
    
    tstep: float
        timestep of time domain signal
    
    title: 
        title of the plot
        
    res: int, optional
        DPI resolution of the figure
        
    linewidth: float, optional
        width of lines in figure
    """
    offset = window_len*0.5
    signal_crop = signal[int(offset):int(((ntraces-1)*window_len+offset))]
    traces = np.split(signal_crop,(ntraces-1))

    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    t_lim = np.floor(1e12*tstep*(window_len-1)/2)+1
    
    #print(t_lim)
    t_ui = (t*1e12)/t_lim
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots()
    
    for i in range(ntraces-1):
        #ax.plot(t*1e12,np.reshape((traces[i][:]),window_len)*1e3, color = 'blue', linewidth = 1)
        ax.plot(t_ui,np.reshape((traces[i][:]),window_len)*1e3, color = 'blue', linewidth = 1)
        
    ax.grid(True,linestyle='--',which='both')
    ax.minorticks_on()
    
    #In picoseconds
    #ax.xaxis.set_major_locator(MultipleLocator(t_lim/2))
    #ax.xaxis.set_minor_locator(MultipleLocator(t_lim/4))
    #ax.xaxis.set_major_formatter('{x:1.0f}')
    
    #In UI
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter('{x:1.1f}')
    
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_major_formatter('{x:1.0f}')
    
    ax.set_xlim([-1, 1])
    #ax.set_xlim([-t_lim, t_lim])
    ax.set_ylim([-300, 300])
    
    plt.xticks(weight='bold',fontsize=20)
    plt.yticks(weight='bold',fontsize=20)
    
    #plt.xlabel('Time (ps)',weight='bold',fontsize=20)
    plt.xlabel('Time (UI)',weight='bold',fontsize=18)
    plt.ylabel('Amplitude (mV)', weight='bold',fontsize=20)


    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    plt.savefig('eye_diagram.pdf',dpi=400)  
    plt.show()
    return True


def pulse_plot(pulse_response, t, samples_per_symbol, n_precursors, n_postcursors, res=1200, title = "Channel Coefficients"):
    #TODO: check for not long enough signal
    #make t optional arg
    """measures and plots channel coefficients from pulse response

    Parameters
    ----------
    pulse_response: array
    
    t: array
        time vector corresponding to pulse response

    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    res : int
        resolution of plot
        
    title: str
        title of plot
    
    Returns
    -------
    channel_coefficients : array
        channel coefficents measured from pulse response

    """
    #number of channel coefficients
    n_cursors = n_precursors + n_postcursors + 1
    channel_coefficients = np.zeros(n_cursors)
    
    #for plotting
    t_vec = np.zeros(n_cursors)
    complete_symbol = round(samples_per_symbol)
    
    #find peak of pulse response = main cursor sample time
    max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]
    
    ui_t = (t - t[max_idx])/(complete_symbol*1e-12) ;

    
    for cursor in range(-n_precursors,n_postcursors+1):
        t_vec[cursor] = t[max_idx + cursor*samples_per_symbol]
        channel_coefficients[cursor] = pulse_response[max_idx + cursor*samples_per_symbol]
    
    
    ui_t_vec = (t_vec - t[max_idx])/(complete_symbol*1e-12) ;
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    #plt.figure(dpi=200)
    fig, ax = plt.subplots()
    
    ax.plot(ui_t_vec, channel_coefficients*1e3, 'o', color = 'blue', linewidth = 1, markersize=10)
    ax.plot(ui_t,pulse_response*1e3, color = 'blue', linewidth = 3)

    plt.xlabel('Unit Interval (UI)',weight='bold',fontsize=16)
    plt.ylabel('Amplitude (mV)', weight='bold',fontsize=16)
    
    plt.xticks(weight='bold',fontsize=16)
    plt.yticks(weight='bold',fontsize=16)
    
    
    ax.grid(True,linestyle='--',which='major',axis = 'y')
    ax.minorticks_on()
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter('{x:0.0f}')
    
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_major_formatter('{x:1.0f}')
    
    ax.set_xlim([-n_precursors, n_postcursors])
    ax.set_ylim([-50, 250])
    
    #plt.legend()
    
    for cursor in range(-n_precursors,n_postcursors):
        plt.axvline(x=cursor+0.5,color = 'grey', linewidth = 0.25, linestyle='--')
        
    plt.show()
    
    #return channel_coefficients