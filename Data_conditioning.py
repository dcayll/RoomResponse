# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:46:12 2020

@author: dcayll

This code will create frequency response curves from time variant data collected 
using electrical, acoustical, and optical data. Data is sampled at 50kHz and is 
in the following format:
    Ch1: Time in seconds
    Ch2: Voltage input to Culvert Amplifier (output of Tektronix AWG)
    Ch3: AC Voltage measured at bias node through 1pF coupling cap and voltage follower
    Ch4: AC Voltage measured at the (+) electrode through a 100:1 capacitive divider referenced to voltage supply gnd
    Ch5: AC Voltage measured at the (-) electrode through a 100:1 capacitive divider referenced to voltage supply gnd
    Ch6: Displacement of center of electrode with Keyence triangulation laser
    Ch7: Trigger from Tektronix AWG (digital 1 when sweep starts, 0 when halfway through, and back to 1 at next sweep)
    Ch8: Audio output from Umik-1 mic (V)


"""

import xarray as xr
import pandas as pd
import sys
from pathlib import Path
import os
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import fftconvolve, convolve
from scipy.io.wavfile import write
import scipy

# from measure import Spectrum
# from room_response_estimator import *
# %matplotlib inline  


def getFileOrg(main_data_path):
    """
    Parameters
    ----------
    main_data_path : pathlib type path
        Path to main folder with subfolders full of data

    Returns
    -------
    DataSeriesNames : dictionary
        Keys are the subfolders in the main data folder in "path", and 
        values are a list the data files in their respective subfolders

    """
    ##### Set path for overall folder with subfolders full of data #####
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200701-electrical, optical, and acoustical measurements')
    
    ##### Determine subfolders in main collection folder #####
    data_directories = [name for name in os.listdir(main_data_path) if os.path.isdir(main_data_path/name)]
    
    DataSeriesNames = {} # dictionary with name of datasubfolders and the filenames of the files in each of them
    
    # iterate through data_directories and fills DataSeriesNames dictionary
    for count, dataSeries in enumerate(data_directories):
        dataPath = main_data_path / dataSeries
        DataSeriesNames.update({dataSeries : [name for name in os.listdir(dataPath) if os.path.splitext(dataPath/name)[1]=='.txt']})
    return DataSeriesNames

def makeDictofDF(dataOrganizationDict, subfolderName):
    '''
    Parameters
    ----------
    dataOrganizationDict : dictionary of lists. 
        keys are subfolders of main data directory;
        values are lists of all files in the subfolders
    subfolderName : String
        name of dataset that is to be processed. 

    Returns
    -------
    dictOfDF : Dictionary of DataFrames
        Puts data from the selected subfolder into "dictOfDF" for organization 
        and later processing

    '''
    dictOfDF = {}
    for count, dataSet in enumerate(dataOrganizationDict.get(subfolderName)):
        # print(dataSet)
        dictOfDF[dataSet[:-4]] = pd.read_csv(main_data_path/subfolderName/dataSet, sep = '\t', header = None)
        dictOfDF.get(dataSet[:-4]).columns = ['Time', 'V_ACbias', 'V_elec+', 'V_elec-', 'D_laser', 'Mic_out']
        # dictOfDF.get(dataSet[:-4]).columns = ['Time', 'V_ACbias', 'V_elec+', 'V_elec-', 'D_laser', 'Mic_out']
        title_metadata = dataSet[:-4].split('_') # turn title into list of strings with dataset information
        
        # populate metadata from title into attrs attribute dictionary
        #   **note if there is an issue with this method, it may be due to removal of .attrs method from pandas since it is experimental
        dictOfDF.get(dataSet[:-4]).attrs['Sample Number'] = title_metadata[0]
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[1]] = float(title_metadata[2])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[3]] = float(title_metadata[4])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[5]+ ' start'] = float(title_metadata[6])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[5]+ ' stop'] = float(title_metadata[7])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[8]+ ' type'] = title_metadata[9]
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[10]] = int(title_metadata[11])
        if len(title_metadata) == 13:
            dictOfDF.get(dataSet[:-4]).attrs['notes'] = title_metadata[12]
        # dictOfDF.get(dataSet[:-4]).attrs[title_metadata[12]+ ' up'] = float(title_metadata[13])
        # dictOfDF.get(dataSet[:-4]).attrs[title_metadata[12]+ ' down'] = float(title_metadata[14])
        # dictOfDF.get(dataSet[:-4]).attrs[title_metadata[15]] = int(title_metadata[16])-1
        # if len(title_metadata) == 13:
        #     dictOfDF.get(dataSet[:-4]).attrs['Run Number'] = int(title_metadata[17])

        print('makeDictofDF {} of {}' .format(count+1, len(dataOrganizationDict.get(subfolderName))))
    return dictOfDF


def getSingleInstance(dictOfDF):
    '''
    Parameters
    ----------
    dictOfDF : Dictionary of DataFrames
        Full dataSet from the selected subfolder with multiple sweeps

    Returns
    -------
    dictOfDF_single : Dictionary of DataFrames
        Reduced dataset of a single sweep according to AWG trigger signal.
        Minimally processed to produce accurate values of collected data 
        from amplifiers/ sensors

    '''
    
    dictOfDF_single = {}
    for count, key in enumerate(dictOfDF):
        # select a single round of the sweep to process, selected by when trigger goes from digital 0 to 1
        fs = dictOfDF.get(key).attrs['fs']
        T = 21.8#dictOfDF.get(key).attrs['duration up']
        startLoc = dictOfDF.get(key).Trigger.diff()[1:int(fs*T)].idxmax(axis=0)
        dictOfDF_single.update({key: dictOfDF.get(key).iloc[startLoc: startLoc + int(fs*T)].reset_index(drop=True)})
        # processing data to meaningful values. time start at 0, electrodes to real V, D_laser to um
        dictOfDF_single.get(key)['Time'] = dictOfDF_single.get(key)['Time']-dictOfDF_single.get(key)['Time'].iloc[0]
        dictOfDF_single.get(key)['V_elec+'] = dictOfDF_single.get(key)['V_elec+'] * 100
        dictOfDF_single.get(key)['V_elec-'] = dictOfDF_single.get(key)['V_elec-'] * 100
        dictOfDF_single.get(key)['D_laser'] = dictOfDF_single.get(key)['D_laser']*1000
        ## add metadata for each sweep depending on filename, key. bias, sample rate, input signal amplitude, freq range
        
        
        
        print('getSingleInstance {} of {}'.format(count+1, len(dictOfDF)))
        
        
    return dictOfDF_single

def makeRevFilters(dictOfDF_single, fs = 50000, low = 20, high = 20000, duration = 5):
    """
    
    Parameters
    ----------
    fs : Int, optional
        Sampling frequency. The default is 50000.
    low : Int, optional
        Starting frequency of sweep. The default is 20.
    high : Int, optional
        Ending frequency of sweep. The default is 20000.
    duration : Int, optional
        Sweep time in seconds. The default is 10.
    dictOfDF_single: dictionary of DataFrames
        Reduced dataset of a single sweep according to AWG trigger signal.
        Minimally processed to produce accurate values of collected data 
        from amplifiers/ sensors

    Returns
    -------
    dictOfDF_single: dictionary of DataFrames
        Contains single sweep data and reverse filter of the input sweep signal

    """
    
        
    
    for count, key in enumerate(dictOfDF_single):
        
        fs = dictOfDF_single.get(key).attrs['fs']
        low = dictOfDF_single.get(key).attrs['freq start']
        high = dictOfDF_single.get(key).attrs['freq stop']
        duration = dictOfDF_single.get(key).attrs['duration up']
        sweepType = dictOfDF_single.get(key).attrs['sweep type']
        T = fs * duration
        w1 = low / fs * 2*np.pi
        w2 = high / fs * 2*np.pi
        # This is what the value of K will be at the end (in dB):
        kend = 10**((-6*np.log2(w2/w1))/20)
        # dB to rational number.
        k = np.log(kend)/T
        
        dictOfDF_single.get(key)['V_input_rev'] = dictOfDF_single.get(key)['V_input'].iloc[::-1].reset_index(drop=True) * \
            np.array(list( map(lambda t: np.exp(float(t)*k), range(int(T)))))
            
        # Now we have to normilze energy of result of dot product.
        # This is "naive" method but it just works.
        Frp =  fft(fftconvolve(dictOfDF_single.get(key)['V_input_rev'], dictOfDF_single.get(key)['V_input']))
        dictOfDF_single.get(key)['V_input_rev'] /= np.abs(Frp[round(Frp.shape[0]/4)])
        print('makeFilters {} of {}'.format(count+1, len(dictOfDF_single)))

    return dictOfDF_single



#%%
if __name__ == '__main__':
    
    ##### Change to path that contains a folder with subfolders full of .txt datafiles #####
    main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200730 - unglued vs glued s10 comparison')
    
    ##### Prompts user for data set desired to process and creates a dictionary of DataFrames full of relevant data #####
    dataOrganizationDict = getFileOrg(main_data_path)
    print('Datasets are grouped in the following subfolders: \n' + ', \n'.join(dataOrganizationDict.keys()))
    subfolderName = input('Please select and type out a dataset to analyze from above: \n\n')
    dictOfDF = makeDictofDF(dataOrganizationDict, subfolderName)
    dictOfDF_single = dictOfDF


    
    # ##### normalize all data #####
    # dictOfDF_norm = normalize(dictOfDF_single)
    # ##### Add timing to data without clear timing signals #####
    # dictOfDF_timed = insertTiming(dictOfDF_norm)
#%% 
    ##### Creates a folder for each of the datasets (entries in dictOfDF) with a .wav file for each channel #####
    # saveWAV(dictOfDF_single, main_data_path, subfolderName)
    # def saveWAV(dictofDF, main_data_path, subfolderName):
        
for count, dataSet in enumerate(dataOrganizationDict.get(subfolderName)):
    os.mkdir(main_data_path/subfolderName/dataSet[:-4])
    # get every dataseries out of the dataframe and normalize them. 
    TargetDir = str(main_data_path/subfolderName/dataSet[:-4])+'\\'
    fs = dictOfDF_single.get(dataSet[:-4]).attrs['fs']
    fs = 48000
    
    #v_input
    V_input_norm = dictOfDF_single.get(dataSet[:-4])['V_input']/dictOfDF_single.get(dataSet[:-4])['V_input'].abs().max()
    write(TargetDir+'V_input_norm.wav', fs, V_input_norm)
    #V_ACbias
    V_ACbias_norm = dictOfDF_single.get(dataSet[:-4])['V_ACbias']/dictOfDF_single.get(dataSet[:-4])['V_ACbias'].abs().max()
    write(TargetDir+'V_ACbias_norm.wav', fs, V_ACbias_norm)
    #V_elec+
    V_elec_p_norm = dictOfDF_single.get(dataSet[:-4])['V_elec+']/dictOfDF_single.get(dataSet[:-4])['V_elec+'].abs().max()
    write(TargetDir+'V_elec_p_norm.wav', fs, V_elec_p_norm)
    #V_elec-
    V_elec_n_norm = dictOfDF_single.get(dataSet[:-4])['V_elec-']/dictOfDF_single.get(dataSet[:-4])['V_elec-'].abs().max()
    write(TargetDir+'V_elec_n_norm.wav', fs, V_elec_n_norm)
    #D_laser
    V_D_laser_norm = dictOfDF_single.get(dataSet[:-4])['D_laser']/dictOfDF_single.get(dataSet[:-4])['D_laser'].abs().max()
    write(TargetDir+'D_laser_norm.wav', fs, V_D_laser_norm)
    #Mic_out
    V_Mic_out_norm = dictOfDF_single.get(dataSet[:-4])['Mic_out']/dictOfDF_single.get(dataSet[:-4])['Mic_out'].abs().max()
    write(TargetDir+'Mic_out_norm.wav', fs, V_Mic_out_norm)
    

    
    #%% for plotting frequency spectra
    
    key = 'S8_Vrms_35.3_bias_600_freq_20_20000_sweep_log_fs_50000_duration_10_0_Nsweeps_4'
    I = fftconvolve( dictOfDF_single.get(key)['V_ACbias'], dictOfDF_single.get(key)['V_input_rev'].iloc[::-1].reset_index(drop=True), mode = 'full')
    I = I[dictOfDF_single.get(key)['V_input_rev'].shape[0]:dictOfDF_single.get(key)['V_input_rev'].shape[0]*2+1]
    Ifft = fft(I)
    # x = np.logspace(0,np.log10(25000), Ifft.size//2)
    x = scipy.fftpack.fftfreq(Ifft.size, 1 / 50e3)
    spectra = plt.figure()
    spectra_ax = spectra.add_subplot(111)
    spectra_ax.plot(x[:x.size//2], abs(Ifft)[:Ifft.size//2])
    # plt.plot(x, abs(Ifft)[:Ifft.size//2])
    spectra_ax.set_xscale('log')
    spectra_ax.set_xlabel('Frequency (Hz)')
    spectra_ax.set_ylabel('Relative Amplitude')
    spectra_ax.set_xlim((20, 20000))
    spectra_ax.set_title('V_ACbias for '+key)
    
    
    #%% for plotting laser and bias data on the same plot and mic data separately. 
    # def plotTimeData(dataDict):
        
for count, key in enumerate(dictOfDF_single):
    
    Vbias_D_laserplt = plt.figure(figsize=(12,6), dpi=100)
    V_D_pltax = Vbias_D_laserplt.add_subplot(111)
    V_ACbiasAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'V_ACbias', grid=True, 
                                              title='V_bias and Center Displacement for {}'.format(key), ax = V_D_pltax)
    V_ACbiasAx.set_ylabel('Bias Voltage (V)')
    V_ACbiasAx.set_xlabel('Time (s)')
    D_laserAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'D_laser', grid=True, secondary_y=True, ax = V_D_pltax)
    D_laserAx.set_ylabel('Center Displacement (um)')
    
    
    Mic_outplt = plt.figure(figsize=(12,6), dpi=100)
    Mic_pltax = Mic_outplt.add_subplot(111)
    Mic_outAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'Mic_out', grid=True, 
                                              title='Mic Output for {}'.format(key), ax = Mic_pltax)
    Mic_outAx.set_ylabel('Mic Output (V)')
    Mic_outAx.set_xlabel('Time (s)')



#%% Datasheet data

key = 's10_Vrms_70.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef'
selectData = dictOfDF.get(key)
bias_current = selectData['V_ACbias']/10**7
peak_current = bias_current.max()
rms_current = 



