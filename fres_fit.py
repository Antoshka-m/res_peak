#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:40:33 2020

@author: Anton Malovichko
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import os.path
from scipy.optimize import curve_fit
import matplotlib as mpl


def get_filenames():
    """
    get filenames with graphical interface
        
    Returns
    -------
    file_list : list
        list of chosen filenames
    """
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    file_list=askopenfilenames()
    return file_list

def lorentzian(f, a, f0, c):
    """
    Lorentzian function
    
    Parameters
    --------
    a: float
        half-width of the peak
    f:
        frequency 
    f0: float
        position of the peak
    
        
    Returns
    -------
    Lorentzian function
    """
    return (a / ((f-f0)**2 + a**2) / np.pi)+c


# define limits for plotting and fitting the peak
f_low = 7000
f_high = 10000
# define initial guess of half-width of peak and its position
#p_guess = [1000, 8200]

files_list = get_filenames()
#sns.set()
sns.set()
sns.set_style("whitegrid")
title=input('Choose title of the plot\n')
plt.figure(figsize=(9, 7))
for file in files_list:
    df = pd.read_csv(file)
    df = df.loc[(df['f, Hz']>f_low)&(df['f, Hz']<f_high)]
    # df['PSD, a.u.']=df['PSD, a.u.']-df['PSD, a.u.'].iloc[0]+1E-20
    # fit data 
    x=np.log10(df['f, Hz'])
    y_orig=np.log10(df['PSD, a.u.'])
    y=np.log10(df['PSD, a.u.'])-np.log10(df['PSD, a.u.']).iloc[0]
    a_guess = 1 / (np.pi * max(y))
    f0_guess = sum(x * y) / sum(y)
    p_guess = [a_guess, np.log10(8200), -0.3]
    popt, pcov = curve_fit(lorentzian, x, y, p0 = p_guess)
    a, f0, c0 = popt[0], popt[1], popt[2]
    print('f=', round(10**f0, 0))
    #fig=sns.lineplot(x=df['f, Hz'], y=df['PSD, a.u.'])
    freq_ar = np.log10(np.arange(f_low, f_high, 1))
    
    plt.title(title, fontsize=20)
    
    ax=sns.lineplot(x=10**x, y=10**(y+y_orig.iloc[0]), alpha=0.25)
    sns.scatterplot(10**f0, 10**(lorentzian([f0], a, f0, c0)+y_orig.iloc[0]), color=ax.get_lines()[-1].get_c(), s=100)
    sns.lineplot(x=10**freq_ar, y=10**(lorentzian(freq_ar, a, f0, c0)+y_orig.iloc[0]), alpha=1, linewidth=3, color=ax.get_lines()[-1].get_c())
ax.set_xlabel(xlabel='f, Hz', fontsize=24)
ax.set_ylabel(ylabel='PSD, a.u.', fontsize=24)
ax.tick_params(labelsize=22)
ax.tick_params(which='minor', labelsize=18)
ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.grid(b=True, which='major', linewidth=1.0)
ax.grid(b=True, which='minor', linewidth=0.5)
#plt.xscale('log')
#plt.yscale('log')
#plt.grid()
# fig.set_yscale('log')
#fig.set_xscale('log')