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
#ECG_Time = np.array(ECG['UnixTime']).reshape(-1) # NOT THE SAME AS ABOVE TIME STAMP
ECG_Standard = (ECG_Data - ECG_Data.mean())/ECG_Data.std()
#ProcessingSteps = np.array(ECG['ProcessingSteps']).reshape(-1)

### Breaking Data and Synchrogonized Timestamp ###
Vehicle = TrialData['Vehicle']
list(Vehicle.keys())
can = Vehicle['can']
list(can.keys())
Brake = can['CarSignals0x760_Brakes']
list(Brake.keys())
Break_Data = np.array(Brake['Data']).reshape(-1)
Break_SyncTime = np.array(Brake['SyncUnixTime']).reshape(-1)
#Break_Time = np.array(Brake['UnixTime']).reshape(-1) #SAME AS ABOVE TIME STAMPS

### Breaking vs Not Breaking Times ### 
Break_ind = np.where(Break_Data == 1)[0]
NoBreak_ind = np.where(Break_Data == 0)[0]
Break_Times = Break_SyncTime[Break_ind]
NoBreak_Times = Break_SyncTime[NoBreak_ind]

### Find Consecutive indices where breaking occurs ###
def consecutiveRanges(a, n):
    length = 1
    list = []
    # If the array is empty,
    # return the list
    if (n == 0):
        return list
    # Traverse the array
    # from first position
    for i in range (1, n + 1):
        # Check the difference
        # between the current
        # and the previous elements
        # If the difference doesn't
        # equal to 1 just increment
        # the length variable.
        if (i == n or a[i] -
            a[i - 1] != 1):
            # If the range contains
            # only one element.
            # add it into the list.
            if (length == 1):
                list.append(str(a[i - length]))
            else:
                # Build the range between the first
                # element of the range and the
                # current previous element as the
                # last range.
                lower = (str(a[i - length]))
                upper = (str(a[i - 1]))
                #temp = (str(a[i - length]) +
                #        " -> " + str(a[i - 1]))
                list.append(lower + " " + upper)
            # After finding the 
            # first range initialize
            # the length by 1 to
            # build the next range.
            length = 1   
        else:
            length += 1
    return list

n_Break_ind = len(Break_ind)
Break_intervals = consecutiveRanges(Break_ind,n_Break_ind) #int(Break_intervals[0][0:5]) = 30425, int(Break_intervals[0][6:11]) = 30551 to split up intervals
n_NoBreak_ind = len(NoBreak_ind)
NoBreak_intervals = consecutiveRanges(NoBreak_ind,n_NoBreak_ind)
Break_intervals.remove('46904') #removing single range values 
NoBreak_intervals.remove('46905') #removing single range values 
NoBreak_intervals[0] = '00000 30424' # fix 0 at start of no breaking

### Find Timestamps for each interval above to create time ranges 
### These time ranges will be used to extract the ECG data from ECG timestamps
### This way we find ECG data in the breaking/nobreaking ranges 
# in first range using the first index value [0:5] to find time: Break_SyncTime[int(Break_intervals[0][0:5])]
# in first range using the second index value [6:11] to find time: Break_SyncTime[int(Break_intervals[0][6:11])]

n_Break_intervals = len(Break_intervals)
break_int_list = [None] * n_Break_intervals
for i in range(n_Break_intervals):
    break_int_list[i] = np.where(np.logical_and(ECG_SyncTime >= Break_SyncTime[int(Break_intervals[i][0:5])], ECG_SyncTime <= Break_SyncTime[int(Break_intervals[i][6:11])]))[0]
ECG_Breaking_Times_Indexes = np.concatenate(break_int_list, axis=0)

##############################################################################
ECG_Breaking_Data = ECG_Data[ECG_Breaking_Times_Indexes]
#ECG_Breaking_SyncTime = ECG_SyncTime[ECG_Breaking_Times_Indexes]
##############################################################################

n_NoBreak_intervals = len(Break_intervals)
nobreak_int_list = [None] * n_NoBreak_intervals
for i in range(n_NoBreak_intervals):
    nobreak_int_list[i] = np.where(np.logical_and(ECG_SyncTime >= Break_SyncTime[int(NoBreak_intervals[i][0:5])], ECG_SyncTime <= Break_SyncTime[int(NoBreak_intervals[i][6:11])]))[0]
ECG_NoBreaking_Times_Indexes = np.concatenate(nobreak_int_list, axis=0)

##############################################################################
ECG_NoBreaking_Data = ECG_Data[ECG_NoBreaking_Times_Indexes]
#ECG_NoBreaking_SyncTime = ECG_SyncTime[ECG_NoBreaking_Times_Indexes]
##############################################################################


##############################################################################################################################################################

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

##############################################################################
#Breaking
##############################################################################
ECG_Breaking_Standard = (ECG_Breaking_Data - ECG_Breaking_Data.mean())/ECG_Breaking_Data.std()

### HRV Analysis and Results ###
#Estimate Sample Rate from Sync Timestamp 
sample_rate = hp.get_samplerate_mstimer(ECG_SyncTime) 
#Applying the filter 
Breaking_filtered = filter_and_visualize(ECG_Breaking_Standard, sample_rate)
Breaking_resampled_signal = resample(Breaking_filtered, len(Breaking_filtered) * 4)
Breaking_wd, Breaking_m = hp.process(hp.scale_data(Breaking_resampled_signal), sample_rate * 4)

plt.figure(figsize=(12,4))
hp.plotter(Breaking_wd, Breaking_m)

for measure in Breaking_m.keys():
    print('%s: %f' %(measure, Breaking_m[measure]))
    
# Breathing signal and Breathing rate plots ###
hp.visualizeutils.plot_breathing(Breaking_wd, Breaking_m)

#plot poincare
plot_poincare(Breaking_wd, Breaking_m, show = True, figsize=None, title='Poincare plot')

##############################################################################
#NoBreaking
##############################################################################
ECG_NoBreaking_Standard = (ECG_NoBreaking_Data - ECG_NoBreaking_Data.mean())/ECG_NoBreaking_Data.std()

### HRV Analysis and Results ###
#Estimate Sample Rate from Sync Timestamp 
sample_rate = hp.get_samplerate_mstimer(ECG_SyncTime) 
#Applying the filter 
NoBreaking_filtered = filter_and_visualize(ECG_NoBreaking_Standard, sample_rate)
NoBreaking_resampled_signal = resample(NoBreaking_filtered, len(NoBreaking_filtered) * 4)
NoBreaking_wd, NoBreaking_m = hp.process(hp.scale_data(NoBreaking_resampled_signal), sample_rate * 4)

plt.figure(figsize=(12,4))
hp.plotter(NoBreaking_wd, NoBreaking_m)

for measure in NoBreaking_m.keys():
    print('%s: %f' %(measure, NoBreaking_m[measure]))
    
# Breathing signal and Breathing rate plots ###
hp.visualizeutils.plot_breathing(NoBreaking_wd, NoBreaking_m)

#plot poincare
plot_poincare(NoBreaking_wd, NoBreaking_m, show = True, figsize=None, title='Poincare plot')
