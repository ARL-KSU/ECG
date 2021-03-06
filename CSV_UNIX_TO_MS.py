# -*- coding: utf-8 -*-
"""
@author: Martin Brown
"""

### Importing TrialData.mat into Python Environment ### 
import h5py
import numpy as np

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
ECG_Data = np.array(ECG['Data'])
ECG_SyncTime = np.array(ECG['SyncUnixTime']).reshape(-1)
ECG_NonSyncTime = np.array(ECG['UnixTime']).reshape(-1)

### Converting Unix Times to ms (first unix time set to zero)
n = len(ECG_SyncTime)
Time = np.zeros(n) # Empty array to fill with millisecond time
for i in range(n):
    Time[i] = ECG_SyncTime[i]-ECG_SyncTime[0]
Time.shape = (n,1)

### Converting Unix Times to ms (first unix time set to zero)
n1 = len(ECG_SyncTime)
Time1 = np.zeros(n) # Empty array to fill with millisecond time
for i in range(n-1):
    Time1[i] = ECG_SyncTime[i+1]-ECG_SyncTime[i]
print(Time1.mean()) # 4.065514218205907 average sampling frequency, f

### Covert to csv file 
# Combine ECG data and new time in ms
ECG_P_session_1 = np.concatenate((Time,ECG_Data), axis = 1)
# Covert to dataframe
import pandas as pd 
ECG_P_session_1_df = pd.DataFrame(ECG_P_session_1, columns = ['Time','ECG'])
# Convert to csv
compression_opts = dict(method='zip',
                        archive_name='ECG_P_session_1.csv')  
ECG_P_session_1_df.to_csv('ECG_P_session_1.zip', index=False,
                          compression=compression_opts) 