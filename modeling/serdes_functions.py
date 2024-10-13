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
    
    for i in range(pattern_zo):
        if pattern_zo[i]>0:
            pattern[i] = 1
        else:
            pattern[i] = 0

    
    return pattern
    