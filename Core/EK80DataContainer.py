import numpy as np
import json

from Core.FIL1 import FIL1


class EK80DataContainer:

    # Should we use pyecholab to get the data from .raw files?
    def __init__(self, jsonfname=None):

        # Constants

        self.z_trd = 75  # (Ohm) Transducer impedance
        self.f_s = 1.5e6  # (Hz) Orginal WBT sampling rate
        self.n_f_points = 1000  # Number of frequency points for evaluation of TS(f) and Sv(f)
        self.noFreq = 1000

        self.hasData = False
        if jsonfname is not None:

            self.jdict = None
            with open(jsonfname,'r') as fp:
                self.jdict = json.load(fp)

            self.parseEK80JSON(self.jdict)

    def parseEK80JSON(self, jdict) :
        xml0 = jdict['XML0']
        self.z_rx_e = xml0['Transceiver']['z_rx_e']
        self.f0 = xml0['Parameter']['FrequencyStart']
        self.fnom = xml0['Transducer']['Frequency']  # nominal design frequency for the transducer
        self.f1 = xml0['Parameter']['FrequencyEnd']

        self.tau = xml0['Parameter']['PulseDuration']
        self.slope = xml0['Parameter']['Slope']
        self.sampleInterval = xml0['Parameter']['SampleInterval']
        self.c = xml0['Environment']['SoundSpeed']
        self.ptx = xml0['Parameter']['TransmitPower']
        self.G_fnom = xml0['Transducer']['GainNom']
        self.PSI_fnom = xml0['Transducer']['EquivalentBeamAngle']
        self.angle_offset_alongship_fnom=xml0['Transducer']['AngleOffsetAlongship']
        self.angle_offset_athwartship_fnom=xml0['Transducer']['AngleOffsetAthwartship']
        self.angle_sensitivity_alongship_fnom=xml0['Transducer']['AngleSensitivityAlongship']
        self.angle_sensitivity_athwartship_fnom=xml0['Transducer']['AngleSensitivityAthwartship']
        self.beam_width_alongship_fnom=xml0['Transducer']['BeamWidthAlongship']
        self.beam_width_athwartship_fnom=xml0['Transducer']['BeamWidthAthwartship']
        self.corrSa = xml0['Transducer']['SaCorrection']
        self.alpha = xml0['Environment']['Alpha']
        self.temperature = xml0['Environment']['Temperature']
        self.salinity = xml0['Environment']['Salinity']
        self.acidity = xml0['Environment']['Acidity']
        self.latitude = xml0['Environment']['Latitude']
        self.depth = xml0['Environment']['Depth']

        self.dropKeelOffset = xml0['Environment']['DropKeelOffset']

        # Test if broadband calibration values exists, if not use nominal values and extrapolate
        if xml0['FrequencyPar']['frequencies']:
            print("Broadband calibration values exists")
            self.frequencies = xml0['FrequencyPar']['frequencies']
            self.gain = xml0['FrequencyPar']['gain']
            self.angle_offset_athwartship = xml0['FrequencyPar']['angle_offset_alongship']
            self.angle_offset_alongship = xml0['FrequencyPar']['angle_offset_alongship']
            self.beam_width_athwartship = xml0['FrequencyPar']['beam_width_athwartship']
            self.beam_width_alongship = xml0['FrequencyPar']['beam_width_alongship']
            self.isCalibrated = True
        elif not xml0['FrequencyPar']['frequencies']:
            print("Broadband calibration values does not exist - use nominal and fit function")
            self.frequencies = None
            self.gain = None
            self.angle_offset_athwartship = None
            self.angle_offset_alongship = None
            self.beam_width_athwartship = None
            self.beam_width_alongship = None
            self.isCalibrated = False
            
        self.fil1s = None
        if 'FIL1' in jdict and 'NaN' not in jdict['FIL1']:
            self.fil1s = []
            for v in jdict['FIL1'].values():
                c = v['coefficients']
                coefficients = np.array(c['real']) + np.array(c['imag']) * 1j
                self.fil1s.append(FIL1(v['noOfCoefficients'], v['decimationFactor'], coefficients))

        raw3 = jdict['RAW3']
        self.offset = raw3['offset']
        self.sampleCount = raw3['sampleCount']
        self.y_rx_nu = None
        if 'quadrant_signals' in raw3 and 'NaN' not in raw3['quadrant_signals']:
            self.y_rx_nu = []
            for v in raw3['quadrant_signals'].values():
                self.y_rx_nu.append(np.array(v['real']) + np.array(v['imag']) * 1j)

            self.y_rx_nu = np.array(self.y_rx_nu)
