import random
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter import filedialog
import csv
import pandas as pd


#You dont need to understand this, just know that it takes a list of numbers, and a desired value. It will return the closest value in the list to K
def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

#open a dialog box and get the selected csv file
root = Tk()
root.withdraw()
csvfile = filedialog.askopenfile(parent=root,mode='r',filetypes=[('Excel file','*.csv')],title='Choose CSV file')

# open another dialog box and get the selected tsv file
tsvfile = filedialog.askopenfile(parent=root,mode='r',filetypes=[('Excel file','*.tsv')],title='Choose TSV file')

##############################################################################
### Driver begins accelerating at an aggressive rate start and end index
##############################################################################

# What kind of events do you want? These should be the EXACT text of what is in the tsv column
eventtype_start = 'Event/Description/Driver begins accelerating at an aggressive rate'
eventtype_end = 'Event/Description/Driver ends accelerating at an aggressive rate'
eventtimes_start = [] # create a blank list of event times
eventtimes_end = [] # create a blank list of event times
f=(1000/4.065514218205907) # Zephr Bioharness 3 sample frequency
                           # 4.06.... average sample rate in CSV_UNIX_TO_MS.py file

# open the selected tsv file
with open(tsvfile.name) as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"') # read the file, line by line, into seperate rows
    for row in rd: # for each row in the tsv file
        data = row[2].split(',') # this 'splits' our row string by commas and creates a list of elements
        if eventtype_start in data[1]: # this checks to see if our event type string (eventtype above) is inside of current tsv file event
            eventtimes_start.append(float(row[1])) # if its the kind of event we want, lets add it to our eventtimes list
        if eventtype_end in data[1]:
            eventtimes_end.append(float(row[1]))
            
df = pd.read_csv(csvfile.name) #ecg data file 
times = df.Time.tolist()
print('The Total Number of Events = ' + str(len(eventtimes_start)))
print('The Total Number of Events = ' + str(len(eventtimes_end)))
#print (eventtimes)

StartIndex = []
EndIndex = [] 
for eventtime in eventtimes_start: # go through each event start time, one at a time
    EventStartTime = eventtime * 1000  # Convert to Milliseconds
    #EventEndTime = (eventtime) * 1000  # no need to Add .650 seconds for p300 then convert to Milliseconds 

    # with our new times, lets find the closest times in our csv file
    StartIndex.append(round(EventStartTime/(1000/f))) 
    
for eventtime in eventtimes_end: # go through each event start time, one at a time
    EventEndTime = eventtime * 1000  # Convert to Milliseconds
    #EventEndTime = (eventtime) * 1000  # no need to Add .650 seconds for p300 then convert to Milliseconds 

    # with our new times, lets find the closest times in our csv file
    EndIndex.append(round(EventEndTime/(1000/f))) 

# Last "Driver begins accelerating at an aggressive rate" has no end, so removed
#StartIndex = StartIndex[:-1]

##############################################################################
# Using the start and end index to run RR interval analysis
##############################################################################

import numpy as np 
import matplotlib.pyplot as plt
import heartpy as hp
from scipy.signal import resample

### Data to numpy array
ECG_Data = df.to_numpy(copy=False)

### Min-Max Normalization 
v = ECG_Data[:, 1]   
ECG_Data[:, 1] = (v - v.min()) / (v.max() - v.min())

### Extract the acceleration events from start index (end indexes not included as some events are too short)
### For this, I will include 5 seconds after the event start index
### 5 s = 5000 ms ; sample rate = 4.065514218205907 ; 5000 / 4.065514218205907 = ceiling(1229.8567245465144) = 1230 instances after start index 
ECG_Agg_Acc = []
for i in range(len(StartIndex)):
    ECG_Agg_Acc.append(ECG_Data[range(StartIndex[i],StartIndex[i]+1230),:])

##############################################################################
# Extracting ibi and bpm from aggressive acceleration in session 1
##############################################################################

### Empty arrays 
ibi = np.zeros(len(ECG_Agg_Acc))
bpm = np.zeros(len(ECG_Agg_Acc))
sample_rate = np.zeros(len(ECG_Agg_Acc))
filtered = []

### Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
from scipy.signal import find_peaks
for i in range(len(ECG_Agg_Acc)):
    # Find sample and remove baseline wander and plot
    sample_rate[i] = hp.get_samplerate_mstimer(ECG_Agg_Acc[i][:,0])
    filtered.append(hp.remove_baseline_wander(ECG_Agg_Acc[i][:,1], sample_rate[i]))
    #plt.figure(figsize=(12,3))
    #plt.title('Signal with Baseline Wander Removed')
    #plt.plot(filtered[i])
    #plt.show()
    
    # Finding peak / rr list 
    peaks, _ = find_peaks(filtered[i], height=0, distance = 150) #consider maxima above 0
                                                                 #positions of QRS complex within the ECG by demanding a distance of at least 150 samples
    # Plot with peaks of R in ECG (results show some misinterpreted peaks but negligible)
    #x = filtered[i]
    #plt.plot(x)
    #plt.plot(peaks, x[peaks], "x")
    #plt.plot(np.zeros_like(x), "--", color="gray")
    #plt.show()
    
    # Find the time (ms) difference between each R wave = ibi = interbeat interval
    #RR_list = peaks.copy() #instances of where rr peaks occur
    ibi[i] = np.diff(ECG_Agg_Acc[i][:,0][peaks]).mean() 
    bpm[i] = 60000 / ibi[i]

np.savetxt("ibi_aggressive_acceleration_session_1.csv", ibi, delimiter=",")
np.savetxt("bpm_aggressive_acceleration_session_1.csv", bpm, delimiter=",")




