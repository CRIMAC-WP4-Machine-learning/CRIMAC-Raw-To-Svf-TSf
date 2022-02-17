# -*- coding: utf-8 -*-
"""
Optional file which allows the reading of EK80 raw data with pyecholab, selecting  a single ping and storing
1.Environmental and calibration data
2.Complex values per sector
in json data file which can be read with the EK80 processing scripts.
Requires that pyecholab is installed
https://github.com/CI-CMG/pyEcholab

Usage:
python readpyecholabPing --channel 2 --pingNo 11 --inputFile E:/IMR-CRIMAC-EK80-FM-38kHz-WC22-WC381.raw --outputFile ..\Data\pyEcholabEK80data.json
or if you are within Spyder or similar:
run readpyecholabPing --channel 2 --pingNo 11 --inputFile E:/IMR-CRIMAC-EK80-FM-38kHz-WC22-WC381.raw --outputFile ..\Data\pyEcholabEK80data.json
Reads channel 2, in this case 38 kHz, and ping number 11 in the raw file
run readpyecholabPing --channel 2 --pingNo 11 --inputFile C:/Users/a32685/Documents/Projects/2020_CRIMAC/CRIMACHackEx/cal-babak-D20201120-T080856.raw
GP 01.06.2021
"""

import json
import numpy as np
from echolab2.instruments import EK80
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show

###############################################################################
# Parsing input
###############################################################################

parser = argparse.ArgumentParser(description='Read Examples raw file and output json data file for a single ping.')
parser.add_argument('--channel', metavar='Channel', type=int,
                    help='Examples channel (transducer) - (Python numbering starts at 0)')
parser.add_argument('--pingNo', metavar='PingNo', type=int,
                    help='Examples ping nummer ')
parser.add_argument('--inputFile', metavar='File',type=str,
                    help='Path and filename of Examples raw file')
parser.add_argument('--outputFile', metavar='outFile',type=str,
                    help='Path and filename of ouput json file')
args = parser.parse_args()

print('File: ',args.inputFile)
print('ChannelNo. :',args.channel)
print('PingNo. :',args.pingNo)

rawfiles = args.inputFile
Channel = args.channel
PingNo = args.pingNo

Channel = Channel-1            # Channel to python indexing (starting at 0)
PingNo = PingNo-1              # PingNo to python indexing (starting at 0)

###############################################################################
# Initialise
###############################################################################
ek80 = EK80.EK80()
ek80.read_raw(rawfiles)
ek80.__dict__

# read raw data from specified channel
raw_list = ek80.raw_data[ek80.channel_ids[Channel]]
raw_data = raw_list[0]
Type=raw_data.configuration[0]['transceiver_type']
Frequency=raw_data.configuration[0]['transducer_frequency']
filenameRaw=raw_data.configuration[0]['file_name']

# get calibration values
calibration = raw_data.get_calibration()
calibration_dict = raw_data.configuration[PingNo]
calibration_fm = calibration_dict['transducer_params_wideband']

# calculate Sv
cal_obj = raw_data.get_calibration()
Sv = raw_data.get_Sv(calibation=cal_obj)

# function to get list of keys (frequencies) for the calibration values
def getList(dict):
    return dict.keys()

keysList = getList(calibration_fm)
cal_fm_freqs=np.ndarray.tolist(np.array(list(keysList)))

gain=[]
angle_offset_athwartship=[]
angle_offset_alongship=[]
beam_width_athwartship=[]
beam_width_alongship=[]

for j in range(0,len(cal_fm_freqs)):
    gain.append(calibration_fm[cal_fm_freqs[j]]['gain'])
    angle_offset_athwartship.append(calibration_fm[cal_fm_freqs[j]]['angle_offset_athwartship'])
    angle_offset_alongship.append(calibration_fm[cal_fm_freqs[j]]['angle_offset_alongship'])
    beam_width_athwartship.append(calibration_fm[cal_fm_freqs[j]]['beam_width_athwartship'])
    beam_width_alongship.append(calibration_fm[cal_fm_freqs[j]]['beam_width_alongship'])

# Complex values
ek80_complex=raw_data.complex

#noCoeff1=r[1]['n_coefficients']
r=calibration.filters

###############################################################################
# XML0 - Nominal values
###############################################################################
fs=calibration.default_sampling_frequency[Type]
FrequencyStart=np.ndarray.tolist(np.array(raw_data.frequency_start[PingNo]))
FrequencyEnd=np.ndarray.tolist(np.array(raw_data.frequency_end[PingNo]))
PulseDurationPing=np.ndarray.tolist(np.array(raw_data.pulse_duration[PingNo]))
SampleInterval=np.ndarray.tolist(np.array(raw_data.sample_interval[PingNo]))
Slope=np.ndarray.tolist(np.array(raw_data.slope[PingNo]))
TransmitPower=np.int(raw_data.transmit_power[PingNo])
Temperature=np.ndarray.tolist(np.array(calibration.temperature))
Salinity=np.ndarray.tolist(np.array(calibration.salinity))
Alpha=np.ndarray.tolist(np.array(calibration.absorption_coefficient[PingNo]))
SoundSpeed=np.ndarray.tolist(np.array(calibration.sound_speed))
#Gain=np.ndarray.tolist(np.array(calibration.gain[PingNo]))
# Using the same pulse duration as in the file (FM)
#GainNom=np.ndarray.tolist(np.array(calibration_dict['gain'][Channel])) # Correct now, but check again
# Changed to using the longest pulse duration
GainNom=np.ndarray.tolist(np.array(calibration_dict['gain'][(len(calibration_dict['gain'])-1)])) # Correct now, but check again
#EquivalentBeamAngle=np.ndarray.tolist(np.array(calibration.equivalent_beam_angle[PingNo]))
EquivalentBeamAngle=np.ndarray.tolist(np.array(calibration_dict['equivalent_beam_angle']))
SaCorrection=np.ndarray.tolist(np.array(calibration.sa_correction[PingNo]))
PingTime=str(np.array(raw_data.ping_time[PingNo]))+'Z'
dropKeelOffset=str(calibration.drop_keel_offset)
Ztransducer=str(raw_data.ZTRANSDUCER)
#Ztranciever=str(raw_data.ZTRANSCEIVER)
Zrecieve=str(cal_obj.impedance)
MaxTXPowerTransducer=(calibration_dict['max_tx_power_transducer'])
PulseDuration=str(calibration_dict['pulse_duration'])
PulseDurationFM=str(calibration_dict['pulse_duration_fm'])
Gain=str(calibration_dict['gain'])
SaCorrection=str(calibration_dict['sa_correction'])
AngleOffsetAlongship=calibration_dict['angle_offset_alongship']
AngleOffsetAthwartship=calibration_dict['angle_offset_athwartship']
AngleSensitivityAlongship=calibration_dict['angle_sensitivity_alongship']
AngleSensitivityAthwartship=calibration_dict['angle_sensitivity_athwartship']
BeamWidthAlongship=calibration_dict['beam_width_alongship']
BeamWidthAthwartship=calibration_dict['beam_width_athwartship']
Depth=np.ndarray.tolist(np.array(calibration.depth))
Acidity=np.ndarray.tolist(np.array(calibration.acidity))
Latitude=np.ndarray.tolist(np.array(cal_obj.latitude))
###############################################################################
# Filters
###############################################################################
# Filter1
fil=raw_data.filters[PingNo]
filt1_real=np.ndarray.tolist(np.real(fil[1]['coefficients']))
filt1_imag=np.ndarray.tolist(np.imag(fil[1]['coefficients']))
noOfCoefficients1=fil[1]['n_coefficients']
decimationFactor1=fil[1]['decimation_factor']
#Filter 2
filt2_real=np.ndarray.tolist(np.real(fil[2]['coefficients']))
filt2_imag=np.ndarray.tolist(np.imag(fil[2]['coefficients']))
noOfCoefficients2=fil[2]['n_coefficients']
decimationFactor2=fil[2]['decimation_factor']

###############################################################################
# RAW3
###############################################################################
SampleCount=np.ndarray.tolist(np.array(raw_data.sample_count[PingNo]))
sampleOffset=np.ndarray.tolist(np.array(raw_data.sample_offset[PingNo])) # Sample offset for PingNo

Q1_real=np.ndarray.tolist(np.real(raw_data.complex[PingNo][:][:,0]))
Q1_imag=np.ndarray.tolist(np.imag(raw_data.complex[PingNo][:][:,0]))
Q2_real=np.ndarray.tolist(np.real(raw_data.complex[PingNo][:][:,1]))
Q2_imag=np.ndarray.tolist(np.imag(raw_data.complex[PingNo][:][:,1]))
Q3_real=np.ndarray.tolist(np.real(raw_data.complex[PingNo][:][:,2]))
Q3_imag=np.ndarray.tolist(np.imag(raw_data.complex[PingNo][:][:,2]))
Q4_real=np.ndarray.tolist(np.real(raw_data.complex[PingNo][:][:,3]))
Q4_imag=np.ndarray.tolist(np.imag(raw_data.complex[PingNo][:][:,3]))


###############################################################################
# RAW3
###############################################################################
plt.plot(Sv.data[10,:])
plt.title('pyecholab - Sv')
plt.grid()
plt.ylim(-150, 10)
plt.xlabel('Sample')
plt.ylabel('Sv')
show()

###############################################################################
# Dict for jsonwrite
###############################################################################
data={
        'COMMENTS': {
                "Free text": "See Readme.txt",
                "Raw file": args.inputFile,
                "json file": args.outputFile
                },
                
        'XML0': {
                'Transceiver': {
                "fs": fs,
                "z_rx_e": float(Zrecieve),
                },
                
                'Channels': {
                "MaxTXPowerTransducer": MaxTXPowerTransducer,
                "PulseDuration": PulseDuration,
                "PulseDurationFM": PulseDurationFM,
                    },
                
                'Transducer': {
                "Frequency": Frequency,
                "FrequencyStart": FrequencyStart,
                "FrequencyEnd": FrequencyEnd,
                "MaxTXPowerTransducer": MaxTXPowerTransducer,
                "Gain": Gain,
                "SaCorrection": SaCorrection,
                "EquivalentBeamAngle": EquivalentBeamAngle,
                "AngleOffsetAlongship": AngleOffsetAlongship,
                "AngleOffsetAthwartship": AngleOffsetAthwartship,
                "AngleSensitivityAlongship": AngleSensitivityAlongship,
                "AngleSensitivityAthwartship": AngleSensitivityAthwartship,
                "BeamWidthAlongship": BeamWidthAlongship,
                "BeamWidthAthwartship": BeamWidthAthwartship,
                "GainNom": GainNom,
                "Nu": 4,
                "z_trd": float(Ztransducer),
                    },
                
                'FrequencyPar': {
                    "frequencies": cal_fm_freqs,
                    "gain": gain,
                    "angle_offset_athwartship": angle_offset_athwartship,
                    "angle_offset_alongship": angle_offset_alongship,
                    "beam_width_athwartship": beam_width_athwartship,
                    "beam_width_alongship": beam_width_alongship,
                    "drop_keel_offset": calibration.drop_keel_offset,
                    },
                
                'Environment': {
                    "Temperature": Temperature,
                    "Salinity": Salinity,
                    "Alpha": Alpha,
                    "SoundSpeed": SoundSpeed,
                    "DropKeelOffset": float(dropKeelOffset),
                    "Depth": float(Depth),
                    "Acidity": float(Acidity),
                    "Latitude": float(Latitude),
                    },

                'Parameter': {
                    "FrequencyStart": FrequencyStart,
                    "FrequencyEnd": FrequencyEnd,
                    "PulseDuration": PulseDurationPing,
                    "SampleInterval": SampleInterval,
                    "Slope": Slope,
                    "TransmitPower": TransmitPower,
                    "PingNo": PingNo+1,
                    "PingTime": PingTime,
                    },
                
                },
                
        'FIL1': {
                '1': {
                "noOfCoefficients": noOfCoefficients1,
                "decimationFactor": decimationFactor1,
                "coefficients": {
                        "real": filt1_real,
                        "imag": filt1_imag
                      }  
                        },
                '2': {
                "noOfCoefficients": noOfCoefficients2,
                "decimationFactor": decimationFactor2,
                "coefficients": {
                        "real": filt2_real,
                        "imag": filt2_imag
                      }  
                }
                },
                
        'RAW3': {
                "offset": sampleOffset,
                "sampleCount": SampleCount,
                'quadrant_signals': {
                        "Q1" :{
                                "real": Q1_real,
                                "imag": Q1_imag
                                },
                        "Q2" :{
                                "real": Q2_real,
                                "imag": Q2_imag
                                },
                        'Q3' :{
                                "real": Q3_real,
                                "imag": Q3_imag
                                },
                        'Q4' :{
                                "real": Q4_real,
                                "imag": Q4_imag
                                }
                        },
                'yc': 'NaN',
                'Sv': 'NaN',
                'Svf': 'NaN',
                'TS': 'NaN',
                'TSf': 'NaN'
                },
                        
}

###############################################################################
# Write to json
###############################################################################
with open(args.outputFile, "w") as outfile:
    json.dump(data, outfile,indent=2)

###############################################################################
# Print some info to screen    
###############################################################################
print('Channel:',raw_data.channel_id)
print('PingNo:',PingNo+1)
