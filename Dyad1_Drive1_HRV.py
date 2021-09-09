# -*- coding: utf-8 -*-
"""
@author: Martin Brown
"""

##############################################################################

### Importing TrialData.mat into Python Environment ### 
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.signal import resample
import heartpy as hp #HRV package

f = h5py.File('TrialData.mat','r')
list(f.keys())
TrialData = f['TrialData']
list(TrialData.keys())

### "Passenger" Data ###
Passenger = TrialData['Passenger']
list(Passenger.keys())
Bioharness = Passenger['Bioharness']
list(Bioharness.keys())

### ECG Data and Synchrogonized Timestamp ###
ECG = Bioharness['ECG']
list(ECG.keys())
ECG_Data = np.array(ECG['Data']).reshape(-1)
ECG_SyncTime = np.array(ECG['SyncUnixTime']).reshape(-1)
ECG_Standard = (ECG_Data - ECG_Data.mean())/ECG_Data.std()

##############################################################################
### https://github.com/paulvangentcom/heartrate_analysis_python/blob/master/examples/2_regular_ECG/Analysing_a_regular_ECG_signal.ipynb ###
##############################################################################
### HRV Analysis Functions ###
### Filtering and Vizualizing ECG ###
def filter_and_visualize(data, sample_rate):
    #function that filters using remove_baseline_wander and visualises result
    filtered = hp.remove_baseline_wander(data, sample_rate)
    plt.figure(figsize=(12,3))
    plt.title('Signal with Baseline Wander Removed')
    plt.plot(filtered)
    plt.show()
    #original signal vs zoomed in corrected signal 
    plt.figure(figsize=(12,3))
    plt.title('Zoomed in Signal with Baseline Wander Removed , Original Signal Overlaid')
    plt.plot(hp.scale_data(data[200:1200]))
    plt.plot(hp.scale_data(filtered[200:1200]))
    plt.show()
    return filtered

### Rotation function for the Poincare ###
def rotate_vec(x, y, angle):
    #rotate vector arounf the origin
    theta = np.radians(angle)
    cs = np.cos(theta)
    sn = np.sin(theta)
    x_rot = (x * cs) - (y * sn)
    y_rot = (x * sn) + (y * cs)
    return x_rot, y_rot

### Poincare Plot Function ###
def plot_poincare(working_data, measures, show = True, figsize=None, title='Poincare plot'): 
    #visualize poincare plot
    #get color palette
    colorpalette = sns.color_palette("bright")
    #get values from dict
    x_plus = working_data['poincare']['x_plus']
    x_minus = working_data['poincare']['x_minus']
    sd1 = measures['sd1']
    sd2 = measures['sd2']
    #define figure
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=figsize)
    #plot scatter
    ax.scatter(x_plus, x_minus, color = colorpalette[0],
                alpha = 0.75, label = 'peak-peak intervals')
    #plot identity line
    mins = np.min([x_plus, x_minus])
    maxs = np.max([x_plus, x_minus])
    identity_line = np.linspace(np.min(mins), np.max(maxs))
    ax.plot(identity_line, identity_line, color='black', alpha=0.5,
             label = 'identity line')
    #rotate SD1, SD2 vectors 45 degrees counterclockwise
    sd1_xrot, sd1_yrot = rotate_vec(0, sd1, 45)
    sd2_xrot, sd2_yrot = rotate_vec(0, sd2, 45)
    #plot rotated SD1, SD2 lines
    ax.plot([np.mean(x_plus), np.mean(x_plus) + sd1_xrot],
             [np.mean(x_minus), np.mean(x_minus) + sd1_yrot],
             color = colorpalette[1], 
             label = 'SD1')
    ax.plot([np.mean(x_plus), np.mean(x_plus) - sd2_xrot],
             [np.mean(x_minus), np.mean(x_minus) + sd2_yrot],
             color = colorpalette[2], 
             label = 'SD2')
    #plot ellipse
    xmn = np.mean(x_plus)
    ymn = np.mean(x_minus)
    el = Ellipse((xmn, ymn), width = sd2 * 2, height = sd1 * 2, angle = 45.0)
    ax.add_artist(el)
    el.set_edgecolor((0,0,0))
    el.fill = False
    ax.set_xlabel('RRi_n (ms)')
    ax.set_ylabel('RRi_n+1 (ms)')
    ax.legend(loc=4, framealpha=0.6)
    ax.set_title(title)
    if show:
        fig.show()
    else:
        return fig

### HRV Analysis and Results ###
#Estimate Sample Rate from Sync Timestamp 
sample_rate = hp.get_samplerate_mstimer(ECG_SyncTime) 
#Applying the filter 
filtered = filter_and_visualize(ECG_Standard, sample_rate)
resampled_signal = resample(filtered, len(filtered) * 4)
wd, m = hp.process(hp.scale_data(resampled_signal), sample_rate * 4)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
    
# Breathing signal and Breathing rate plots ###
hp.visualizeutils.plot_breathing(wd, m)

#plot poincare
plot_poincare(wd, m, show = True, figsize=None, title='Poincare plot')

##############################################################################
### Saving all the HRV Measurements ### 

import csv
with open('Measures_Dyad1_Drive1.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in m.items():
       writer.writerow([key, value])