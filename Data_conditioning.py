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

import pandas as pd
import sys
from pathlib import Path
import os
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import fftconvolve, convolve
from scipy.io.wavfile import write
import scipy
import math

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
        # dictOfDF.get(dataSet[:-4]).columns = ['Time', 'V_input', 'V_ACbias', 'V_elec+', 'V_elec-', 'D_laser', 'Trigger', 'Mic_out']
        # dictOfDF.get(dataSet[:-4]).columns = ['Time', 'V_ACbias', 'V_elec+', 'V_elec-', 'D_laser', 'Mic_out']
        if dictOfDF.get(dataSet[:-4]).columns.size == 2:
            dictOfDF.get(dataSet[:-4]).columns = ['Time', 'D_laser']
        elif dictOfDF.get(dataSet[:-4]).columns.size == 3:
            dictOfDF.get(dataSet[:-4]).columns = ['Time', 'D_laser', 'V_input']

        # dictOfDF.get(dataSet[:-4]).columns = ['Time', 'V_elec+', 'V_elec-', 'V_ACbias', 'Mic_out']
        title_metadata = dataSet[:-4].split('_') # turn title into list of strings with dataset information

        # populate metadata from title into attrs attribute dictionary
        dictOfDF.get(dataSet[:-4]).attrs['Sample Number'] = title_metadata[0]
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[1]] = float(title_metadata[2])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[3]] = float(title_metadata[4])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[5]+ ' start'] = float(title_metadata[6])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[5]+ ' stop'] = float(title_metadata[7])
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[8]+ ' type'] = title_metadata[9]
        dictOfDF.get(dataSet[:-4]).attrs[title_metadata[10]] = int(title_metadata[11])
        # dictOfDF.get(dataSet[:-4]).attrs['Burn-in time (s)'] = int(title_metadata[13])
        # dictOfDF.get(dataSet[:-4]).attrs['Location (mm)'] = int(title_metadata[13])
        # if len(title_metadata) == 15:
        #     dictOfDF.get(dataSet[:-4]).attrs['notes'] = title_metadata[12]
        
        # if len(title_metadata) == 14:
        #     dictOfDF.get(dataSet[:-4]).attrs['notes'] = title_metadata[13]
        
        
        # keyence laser calibration
        # dictOfDF.get(dataSet[:-4])['D_laser'] = (dictOfDF.get(dataSet[:-4])['D_laser']-dictOfDF.get(dataSet[:-4])['D_laser'][0:144001].mean())*1000
        
        
        ## U-E laser calibration
        dictOfDF.get(dataSet[:-4])['D_laser'] = (dictOfDF.get(dataSet[:-4])['D_laser']-dictOfDF.get(dataSet[:-4])['D_laser'][0:144001].mean())*1000 #*.2 #as of 3/8/2021, data is saved in mm already. 
    
    

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
        T = dictOfDF.get(key).attrs['duration up']
        startLoc = dictOfDF.get(key).Trigger.diff()[1:int(fs*T)].idxmax(axis=0)
        dictOfDF_single.update({key: dictOfDF.get(key).iloc[startLoc: startLoc + int(fs*T)].reset_index(drop=True)})
        # processing data to meaningful values. time start at 0, electrodes to real V, D_laser to um
        dictOfDF_single.get(key)['Time'] = dictOfDF_single.get(key)['Time']-dictOfDF_single.get(key)['Time'].iloc[0]
        dictOfDF_single.get(key)['V_elec+'] = dictOfDF_single.get(key)['V_elec+'] * 100
        dictOfDF_single.get(key)['V_elec-'] = dictOfDF_single.get(key)['V_elec-'] * 100
        
        ## U-E laser calibration
        dictOfDF_single.get(key)['D_laser'] = (dictOfDF_single.get(key)['D_laser']-dictOfDF_single.get(key)['D_laser'][0:144001].mean())*.2
    
        ## keyence laser calibration
        # dictOfDF_single.get(key)['D_laser'] = dictOfDF_single.get(key)['D_laser'] * 1000



        print('getSingleInstance {} of {}'.format(count+1, len(dictOfDF)))


    return dictOfDF_single

# def makeRevFilters(dictOfDF_single, fs = 50000, low = 20, high = 20000, duration = 5):
#     """

#     Parameters
#     ----------
#     fs : Int, optional
#         Sampling frequency. The default is 50000.
#     low : Int, optional
#         Starting frequency of sweep. The default is 20.
#     high : Int, optional
#         Ending frequency of sweep. The default is 20000.
#     duration : Int, optional
#         Sweep time in seconds. The default is 10.
#     dictOfDF_single: dictionary of DataFrames
#         Reduced dataset of a single sweep according to AWG trigger signal.
#         Minimally processed to produce accurate values of collected data
#         from amplifiers/ sensors

#     Returns
#     -------
#     dictOfDF_single: dictionary of DataFrames
#         Contains single sweep data and reverse filter of the input sweep signal

#     """



#     for count, key in enumerate(dictOfDF_single):

#         fs = dictOfDF_single.get(key).attrs['fs']
#         low = dictOfDF_single.get(key).attrs['freq start']
#         high = dictOfDF_single.get(key).attrs['freq stop']
#         duration = 30
#         sweepType = 'log'
#         T = fs * duration
#         w1 = low / fs * 2*np.pi
#         w2 = high / fs * 2*np.pi
#         # This is what the value of K will be at the end (in dB):
#         kend = 10**((-6*np.log2(w2/w1))/20)
#         # dB to rational number.
#         k = np.log(kend)/T

#         dictOfDF_single.get(key)['V_input_rev'] = dictOfDF_single.get(key)['V_input'].iloc[::-1].reset_index(drop=True) * \
#             np.array(list( map(lambda t: np.exp(float(t)*k), range(int(T)))))

#         # Now we have to normilze energy of result of dot product.
#         # This is "naive" method but it just works.
#         Frp =  fft(fftconvolve(dictOfDF_single.get(key)['V_input_rev'], dictOfDF_single.get(key)['V_input']))
#         dictOfDF_single.get(key)['V_input_rev'] /= np.abs(Frp[round(Frp.shape[0]/4)])
#         print('makeFilters {} of {}'.format(count+1, len(dictOfDF_single)))

#     return dictOfDF_single



def normalize(dictOfDF_single):
    """


    Parameters
    ----------
    dictOfDF_single : dictionary of DataFrames
        Contains single sweep data raw values in float32

    Returns
    -------
    dictOfDF_norm : dictionary of DataFrames
        Contains single sweep data with normalized float32 values

    """
    for count, dataSet in enumerate(dataOrganizationDict.get(subfolderName)):
        # save normalized
        #V_ACbias
        V_ACbias_norm = dictOfDF_single.get(dataSet[:-4])['V_ACbias']/dictOfDF_single.get(dataSet[:-4])['V_ACbias'].abs().max()
        #V_elec+
        V_elec_p_norm = dictOfDF_single.get(dataSet[:-4])['V_elec+']/dictOfDF_single.get(dataSet[:-4])['V_elec+'].abs().max()
        #V_elec-
        V_elec_n_norm = dictOfDF_single.get(dataSet[:-4])['V_elec-']/dictOfDF_single.get(dataSet[:-4])['V_elec-'].abs().max()
        #D_laser
        D_laser_norm = dictOfDF_single.get(dataSet[:-4])['D_laser']/dictOfDF_single.get(dataSet[:-4])['D_laser'].abs().max()
        #Mic_out
        V_Mic_out_norm = dictOfDF_single.get(dataSet[:-4])['Mic_out']/dictOfDF_single.get(dataSet[:-4])['Mic_out'].abs().max()

        # save normalized
        dictOfDF_single.get(dataSet[:-4])['V_ACbias'] = V_ACbias_norm*.5
        dictOfDF_single.get(dataSet[:-4])['V_elec+'] = V_elec_p_norm*.5
        dictOfDF_single.get(dataSet[:-4])['V_elec-'] = V_elec_n_norm*.5
        dictOfDF_single.get(dataSet[:-4])['D_laser'] = D_laser_norm*.5

    return dictOfDF_single



def saveWAV(dictOfDF_single, main_data_path, subfolderName, dataOrganizationDict, label, timing):
    """


    Parameters
    ----------
    dictofDF_single : dictionary of DataFrames
        Contains single sweep data
    main_data_path : pathlib type path
        Path to main folder with subfolders full of data
    subfolderName : String
        name of dataset that is to be processed.
    dataOrganizationDict : dictionary of lists.
        keys are subfolders of main data directory;
        values are lists of all files in the subfolders

    Returns
    -------
    None.

    """


    for count, dataSet in enumerate(dataOrganizationDict.get(subfolderName)):
        
        # # for creating a unique folder for every dataset: 
        # os.mkdir(main_data_path/subfolderName/dataSet[:-4])
        # TargetDir = str(main_data_path/subfolderName/dataSet[:-4])+'\\'
        
        # for putting .wav files into the same folder as the .txt data. 
        TargetDir = str(main_data_path/subfolderName)+'\\'
        
        fs = dictOfDF_single.get(dataSet[:-4]).attrs['fs']
        # fs = 48000
        
        # #V_input
        # V_input = dictOfDF_single.get(dataSet[:-4])['V_input']
        # write(TargetDir+'V_input.wav', fs, V_input)


        # #V_ACbias
        # V_ACbias_norm = dictOfDF_single.get(dataSet[:-4])['V_ACbias']
        # write(TargetDir+'V_ACbias_norm.wav', fs, V_ACbias_norm)
        # #V_elec+
        # V_elec_p_norm = dictOfDF_single.get(dataSet[:-4])['V_elec+']
        # write(TargetDir+'V_elec_p__{}.wav'.format(label), fs, V_elec_p_norm)
        # #V_elec-
        # V_elec_n_norm = dictOfDF_single.get(dataSet[:-4])['V_elec-']
        # write(TargetDir+'V_elec_n_norm.wav', fs, V_elec_n_norm)
        #D_laser
        D_laser_norm = dictOfDF_single.get(dataSet[:-4])['D_laser']
        write(TargetDir+'D_{}_{}V_{}_{}.wav'.format(label, math.trunc(dictOfDF.get(dataSet[:-4]).attrs.get('Vrms')), 
                                                   math.trunc(dictOfDF.get(dataSet[:-4]).attrs.get('bias')), timing), fs, D_laser_norm)
        # #Mic_out
        # V_Mic_out_norm = dictOfDF_single.get(dataSet[:-4])['Mic_out']
        # write(TargetDir+'Mic_out_norm_{}.wav'.format(label), fs, V_Mic_out_norm)


def insertTiming(dictOfDF_norm):
    """


    Parameters
    ----------
    dictOfDF_norm : dictionary of DataFrames
        Contains single sweep data with normalized float32 values

    Returns
    -------
    dictOfDF_norm : dictionary of DataFrames
        Contains single sweep data with normalized float32 values
        with timing references added to V_ACbias and D_laser from
        V_elec- series

    """


    for count, key in enumerate(dictOfDF_norm):

        # finds index where the timing signal ends/begins for the front/rear timing signal respectively
        timingSig_index_front = int(dictOfDF_norm.get(key)[dictOfDF_norm.get(key)['V_elec-'].gt(0.3)].index[0]+dictOfDF_norm.get(key).attrs['fs']*.5)
        timingSig_index_back = dictOfDF_norm.get(key)['V_elec-'].shape[0] - int(dictOfDF_norm.get(key)[dictOfDF_norm.get(key)['V_elec-'].iloc[::-1].reset_index(drop=True)
                                                                                                    .gt(0.3)].index[0]+dictOfDF_norm.get(key).attrs['fs']*.5)

        # gets timing signal from V_elec- and copies and pastes it into the other sweeps without significant beginning sweeps
        timingSig_front = dictOfDF_norm.get(key)['V_elec-'][:timingSig_index_front]
        timingSig_back = dictOfDF_norm.get(key)['V_elec-'][timingSig_index_back:]


        dictOfDF_norm.get(key)['V_ACbias'][:timingSig_index_front] = timingSig_front/timingSig_front.abs().max()*.5
        # dictOfDF_norm.get(key)['D_laser'][:timingSig_index_front] = timingSig_front/timingSig_front.abs().max()*.5

        dictOfDF_norm.get(key)['V_ACbias'][timingSig_index_back:] = timingSig_back/timingSig_back.abs().max()*.5
        # dictOfDF_norm.get(key)['D_laser'][timingSig_index_back:] = timingSig_back/timingSig_back.abs().max()*.5



        print('insertTiming {} of {}'.format(count+1, len(dictOfDF_norm)))

    return dictOfDF_norm

def singleInstanceFromTimingRef(dictOfDF):
    """


    Parameters
    ----------
    dictOfDF : TYPE
        DESCRIPTION.

    Returns
    -------
    dictOfDF_NoTiming : Dictionary of DataFrames
        Removes timing reference chirp at beginning of sweep.

    """


    dictOfDF_NoTiming = {}

    for count, key in enumerate(dictOfDF):

        # finds index where the timing signal ends/begins for the front/rear timing signal respectively
        timingSig_index_front = int(dictOfDF.get(key)[dictOfDF.get(key)['V_elec-'].gt(0.3)].index[0])#+dictOfDF.get(key).attrs['fs']*.5)
        timingSig_index_back = dictOfDF.get(key)['V_elec-'].shape[0] - int(dictOfDF.get(key)[dictOfDF.get(key)['V_elec-'].iloc[::-1].reset_index(drop=True)
                                                                                                    .gt(0.3)].index[0])#+dictOfDF.get(key).attrs['fs']*.5)

        #create a dict of df with timing signals removed from beginning and end of the signal.
        dictOfDF_NoTiming[key] = dictOfDF.get(key)[timingSig_index_front:timingSig_index_back].reset_index(drop=True)

        # #find exact location of beginning of sweep ( .gt commands are the cutoff voltages)
        # SweepStart = int(dictOfDF_NoTiming.get(key)[dictOfDF_NoTiming.get(key)['V_elec-'].gt(0.5)].index[0])
        # SweepEnd = dictOfDF_NoTiming.get(key)['V_elec-'].shape[0] - int(dictOfDF_NoTiming.get(key)[dictOfDF_NoTiming.get(key)['V_elec-'].iloc[::-1].reset_index(drop=True)
        #                                                                                             .gt(0.5)].index[0])
        # dictOfDF_NoTiming[key] = dictOfDF_NoTiming.get(key)[SweepStart:SweepEnd].reset_index(drop=True)
        # dictOfDF_NoTiming.get(key)['Time'] = dictOfDF_NoTiming.get(key)['Time'] - dictOfDF_NoTiming.get(key)['Time'][0]




        print('SingleInstanceFromRef {} of {}'.format(count+1, len(dictOfDF)))
    return dictOfDF_NoTiming


def plotTimeDomain(dictOfDF, path, sample, dataSet, voltage):
    """
    

    Parameters
    ----------
    dictOfDF : dictionary of DataFrames
        Contains all sweep data from 
    sample : String
        number of sample that data will be plotted from
    dataSet : String
        D_laser, V_input, or V_input_rev
    voltage : TYPE
        Sweep Voltage (35, 70, 140, 220) 

    Returns
    -------
    plots time domain figure using matplotlib and pandas of the specified dataSet of the given sample at a given voltage

    """
    
    # if len(voltage) == 1:
    #     voltage = [voltage]
    
    for v in voltage:
        
        key = '{}_Vrms_{}_bias_600_freq_20_20000_sweep_log_fs_48000_R'.format(sample, v)
        
        TimeDomain_plot = plt.figure(figsize=(9,5), dpi=100)
        TimeDomain_pltax = TimeDomain_plot.add_subplot(111)
        plt.gcf().subplots_adjust(bottom=0.2)
        
        if dictOfDF.get(key) is None:
            print('breakOut')
            break
        
        TimeDomain_pltax = dictOfDF.get(key).plot(x = 'Time', y = dataSet, label = '{}_{}V'.format(sample, v), 
                                                       grid=True, ax = TimeDomain_pltax)
        if dataSet == 'D_laser':
            TimeDomain_pltax.set_xlabel('Time (s)')
            TimeDomain_pltax.set_ylabel('Displacement (\u03BCm)')
            plt.title('Time Domain Displacement')
        elif dataSet == 'V_input':
            TimeDomain_pltax.set_xlabel('Time (s)')
            TimeDomain_pltax.set_ylabel('Amp input Voltage (V)')
            plt.title('Time Domain Voltage Input')
        elif dataSet == 'V_input_rev':
            TimeDomain_pltax.set_xlabel('Time (s)')
            TimeDomain_pltax.set_ylabel('Voltage (V)')
            plt.title('Time Domain Reverse Filter')
        TimeDomain_plot.savefig(path + '\\' + sample + ' before' + '\\' + sample + '_' + dataSet + ' at ' + str(v) + ' V.png')
        





#%%
if __name__ == '__main__':

    ##### Change to path that contains a folder with subfolders full of .txt datafiles #####

    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200701-electrical, optical, and acoustical measurements')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200729 - Electrical Insulation Measurements')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200805 - open face test, 1V input')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200811 - New Diaphragm Characterization')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200811 - Open Face test parallel, perpendicular')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200820 - Laser tests')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200822 - Pink DLC diaphragm')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200826 - Open face test, real displacement')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200826 - stax diaphragm')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200903 - differential amp measurement')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20200901 - New Stacking fixture')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20201108 - Samsung tests')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20201110 - Sennheiser Driver in free air')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20201118 - Samsung 100hr test 2')
    # main_data_path = Path('G:\\My Drive\\Dynamic Voltage Measurement\\20201202 - 1009-2, 0909-8, 0909-9, 0908-1')
    # main_data_path = Path('G:\\Shared drives\\16mm Coin\\Coin data\\20201202 - 0908-1')
    
    
    #%%  batch creation of time domain plots and .wav files of displacement data
    
# finds all the coins tested on "data_date" and creates a list of the coin ID numbers
    base_path = 'G:\\Shared drives\\16mm Coin\\Coin data'
    data_date = '20210307'
    
# gets all folders of coin data into a single list named "samples"
    samples = os.listdir(base_path)
    if 'desktop.ini' in samples:
        samples.remove('desktop.ini')
    if 'Free air resonance tracking.xlsx' in samples:
        samples.remove('Free air resonance tracking.xlsx')
    if '~$Free air resonance tracking.xlsx' in samples:
        samples.remove('~$Free air resonance tracking.xlsx')
    
# creates a list of the coin data collected on "data_date"
    folderName = []
    CoinDataFrom_data_date = []
    for s in samples:
        folderName.append(s.split(' - '))
        if s.split(' - ')[0] == data_date:
            CoinDataFrom_data_date.append(s.split(' - ')[1])
            
# for geting time domain plots and .wav files of data saved to google drive
    for coin in CoinDataFrom_data_date:
        print(coin)
        totalPath = base_path + '\\' + data_date + ' - ' + coin
        subfolderName = totalPath.split()[-1]+' before'
        main_data_path = Path(totalPath)
        
        dataOrganizationDict = getFileOrg(main_data_path)
        dictOfDF = makeDictofDF(dataOrganizationDict, subfolderName)
        plotTimeDomain(dictOfDF, totalPath, subfolderName.split()[0], 'D_laser', [35, 70, 140, 220])
        plt.close('all')
        
        # saves 'D_laser' data into .wav file for import into REW
        wavName = subfolderName.split()[0]
        if subfolderName.split()[1] == 'before':
            timing = 'b4'
        else:
            timing = 'af'
        saveWAV(dictOfDF, main_data_path, subfolderName, dataOrganizationDict, wavName, timing)
    
    
#%% Single coin's processing
    temp_path = 'G:\\Shared drives\\16mm Coin\\Coin data\\20210307 - 0103-3'
    # temp = 'G:\\Shared drives\\16mm Coin\\Coin data\\20201118 - samsung 2 - 0909-1'
    subfolderName = temp_path.split()[-1]+' before'
    main_data_path = Path(temp_path)


    ##### Prompts user for data set desired to process and creates a dictionary of DataFrames full of relevant data #####
    dataOrganizationDict = getFileOrg(main_data_path)
    # print('Datasets are grouped in the following subfolders: \n' + ', \n'.join(dataOrganizationDict.keys()))
    # subfolderName = input('Please select and type out a dataset to analyze from above: \n\n')
    
    dictOfDF = makeDictofDF(dataOrganizationDict, subfolderName)
    plotTimeDomain(dictOfDF, totalPath, subfolderName.split()[0], 'D_laser', [35, 70, 140, 220])
    
    
    # saves displacment data into a wav file for import into REW
    wavName = subfolderName.split()[0]
    if subfolderName.split()[1] == 'before':
        timing = 'b4'
    else:
        timing = 'af'
    saveWAV(dictOfDF, main_data_path, subfolderName, dataOrganizationDict, wavName, timing)


#%%


    # ##### Downsizes full dataset to a single sweep for each run and starts it at beginning of sweep #####
    # dictOfDF_single = getSingleInstance(dictOfDF)

    # ##### Creates reverse filter and adds it to dictionary of DataFrames #####
    # dictOfDF_revFilt = makeRevFilters(dictOfDF_single)



    #%% for plotting frequency spectra

    key = 's8_Vrms_35.3_bias_600_freq_20_20000_sweep_log_fs_48000_duration_30_0_Nsweeps_1'
    I = fftconvolve( dictOfDF_revFilt.get(key)['V_ACbias'], dictOfDF_revFilt.get(key)['V_input_rev'].iloc[::-1].reset_index(drop=True), mode = 'full')
    I = I[dictOfDF_revFilt.get(key)['V_input_rev'].shape[0]:dictOfDF_revFilt.get(key)['V_input_rev'].shape[0]*2+1]
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


#     #%% for plotting laser and bias data on the same plot and mic data separately.
#     # def plotTimeData(dataDict):

# for count, key in enumerate(dictOfDF):

#     Vbias_D_laserplt = plt.figure(figsize=(12,6), dpi=100)
#     V_D_pltax = Vbias_D_laserplt.add_subplot(111)
    
    
    
#     # V_ACbiasAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'V_ACbias', grid=True,
#     #                                            secondary_y=True, ax = V_D_pltax)
#     # V_ACbiasAx.set_ylabel('AC Bias Voltage (V)')
#     D_laserAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'D_laser', title='disp for {}'.format(key),grid=True, ax = V_D_pltax)
#     D_laserAx.set_xlabel('Time (s)')
#     D_laserAx.set_ylabel('Center Displacement (um)')
    
#     V_inplt = plt.figure(figsize=(12,6), dpi=100)
#     V_in_pltax = V_inplt.add_subplot(111)
#     V_inAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'V_elec-', title='V_input for {}'.format(key),grid=True, ax = V_in_pltax)
#     V_inAx.set_xlabel('Time (s)')
#     V_inAx.set_ylabel('Electrode Voltage (V)')
    
    
#     # Mic_outplt = plt.figure(figsize=(12,6), dpi=100)
#     # Mic_pltax = Mic_outplt.add_subplot(111)
#     # Mic_outAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'Mic_out', grid=True,
#     #                                           title='Mic Output for {}'.format(key), ax = Mic_pltax)
#     # Mic_outAx.set_ylabel('Mic Output (V)')
#     # Mic_outAx.set_xlabel('Time (s)')

# #%% 

# newPlt = plt.figure(figsize=(12,6), dpi=100)
# movAvgAx = newPlt.add_subplot(111)
# D_laserAx = dictOfDF.get(key).plot(x = 'Time', y = 'D_laser', title='Creating "beats" using moving average',grid=True, ax = movAvgAx)
# D_laserAx.set_xlabel('Time (s)')
# D_laserAx.set_ylabel('Center Displacement (mm)')
# plt.plot(dictOfDF.get(key)['Time'], dictOfDF.get(key)['D_laser'].rolling(2048).mean())
# movAvgAx.legend(['Raw Data', '2048 pt. moving average'])

# #%% comparing 3 locations along diaphragm

# # # #Closed faced - center of diaphragm
# # # closed_center = dictOfDF_single.get('s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Closed')
# # #Open faced - center of diaphragm
# # center = dictOfDF_single.get('s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Opencenter')
# # #Open faced - 3mm closer to table
# # ClosertoHole = dictOfDF_single.get('s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Open3mmTowardsTable')
# # #Open faced - 3mm farther from table
# # FarthertoHole = dictOfDF_single.get('s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Open3mmAwayTable')

# # # create dictionary from data from above:
# # openFace = {}
# # # openFace['s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Closed'] = closed_center
# # openFace['s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Opencenter'] = center
# # openFace['s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Open3mmTowardsTable'] = ClosertoHole
# # openFace['s9_Vrms_176.7_bias_600_freq_20_20000_sweep_log_fs_48000_timingRef_Open3mmAwayTable'] = FarthertoHole

# # dictOfDF_timeSync = singleInstanceFromTimingRef(openFace)

# dictOfDF_timeSync = singleInstanceFromTimingRef(dictOfDF)

# D_laserplt = plt.figure(figsize=(12,6), dpi=100)
# V_D_pltax = D_laserplt.add_subplot(111)
# # V_ACbiasAx = dictOfDF_single.get(key).plot(x = 'Time', y = 'V_ACbias', grid=True,
# #                                           title='V_bias and Center Displacement for {}'.format(key), ax = V_D_pltax)
# # V_ACbiasAx.set_ylabel('Bias Voltage (V)')

# for count, key in enumerate(dictOfDF_timeSync):
#     #remove offset in D_laser data and begin time at 0
#     dictOfDF_timeSync.get(key)['D_laser'] = dictOfDF_timeSync.get(key)['D_laser'].sub(dictOfDF_timeSync.get(key)['D_laser'].mean())
#     dictOfDF_timeSync.get(key)['Time'] = dictOfDF_timeSync.get(key)['Time'].sub(dictOfDF_timeSync.get(key)['Time'][0])
    
#     dictOfDF_timeSync.get(key)['D_laser'] = dictOfDF_timeSync.get(key)['D_laser'].rolling(20).mean()
#     dictOfDF_timeSync.get(key)['Time'] = dictOfDF_timeSync.get(key)['Time'].rolling(20).mean()
    
#     D_laserAx = dictOfDF_timeSync.get(key).plot(x = 'Time', y = 'D_laser', grid=True, title='',ax = V_D_pltax,
#                                                 label = dictOfDF_timeSync.get(key).attrs['Location (mm)'])
#     D_laserAx.set_ylabel('Displacement (um)')
#     D_laserAx.set_xlabel('Time (s)')





# #%%

# plt_0324_1_ax = d_70V.plot(x = 'Time', y = 'D_laser', title = '0324-1 70V', legend = False, grid = 'both')
# plt_0324_1_ax.set_ylabel('Displacement (mm)')
# plt_0324_1_ax.set_xlabel('Time (s)')

# plt_0915_1_ax = d_0915_1_70V.plot(x = 'Time', y = 'D_laser', title = '0915-1 70V', legend = False, grid = 'both')
# plt_0915_1_ax.set_ylabel('Displacement (mm)')
# plt_0915_1_ax.set_xlabel('Time (s)')

# plt_0915_2_ax = d_0915_2_70V.plot(x = 'Time', y = 'D_laser', title = '0915-2 70V', legend = False, grid = 'both')
# plt_0915_2_ax.set_ylabel('Displacement (mm)')
# plt_0915_2_ax.set_xlabel('Time (s)')





