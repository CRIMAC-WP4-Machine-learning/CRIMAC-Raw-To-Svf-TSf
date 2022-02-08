import numpy as np
import json

from Core.FIL1 import FIL1


class Derived:
    def __init__(self, frqp, parm, trdu):
        # Calculate Gfc
        if frqp.isCalibrated:
            # Calibrated case
            self.Gfc = np.interp(parm.f_c, frqp.frequencies, frqp.gain)
        else:
            # Uncalibrated case
            self.Gfc = trdu.G_fnom + 20 * np.log10(frqp.frequencies/trdu.fnom)
        
        # Calculate PSI_f
        self.PSI_f = trdu.PSI_fnom + 20 * np.log10(trdu.fnom/frqp.frequencies)



class Constants:
    def __init__(self, z_td_e, f_s, n_f_points):
        self.z_td_e = z_td_e
        self.f_s = f_s
        self.n_f_points = n_f_points


class Transceiver:
    def __init__(self, xml):
        self.z_rx_e = xml['z_rx_e']

class Parameter:
    def __init__(self, xml):
        self.f0 = xml['FrequencyStart']
        self.f1 = xml['FrequencyEnd']
        self.f_c = (self.f0 + self.f1) / 2.0
        self.tau = xml['PulseDuration']
        self.slope = xml['Slope']
        self.sampleInterval = xml['SampleInterval']
        self.ptx = xml['TransmitPower']

class Transducer:
    def __init__(self, xml):
        self.fnom = xml['Frequency']  # nominal design frequency for the transducer
        self.G_fnom = xml['GainNom']
        self.PSI_fnom = xml['EquivalentBeamAngle']
        self.angle_offset_alongship_fnom = xml['AngleOffsetAlongship']
        self.angle_offset_athwartship_fnom = xml['AngleOffsetAthwartship']
        self.angle_sensitivity_alongship_fnom = xml['AngleSensitivityAlongship']
        self.angle_sensitivity_athwartship_fnom = xml['AngleSensitivityAthwartship']
        self.beam_width_alongship_fnom = xml['BeamWidthAlongship']
        self.beam_width_athwartship_fnom = xml['BeamWidthAthwartship']
        self.corrSa = xml['SaCorrection']

class Environment:
    def __init__(self, xml):
        self.c = xml['SoundSpeed']
        self.alpha = xml['Alpha']
        self.temperature = xml['Temperature']
        self.salinity = xml['Salinity']
        self.acidity = xml['Acidity']
        self.latitude = xml['Latitude']
        self.depth = xml['Depth']
        self.dropKeelOffset = xml['DropKeelOffset']

class FrequencyPar:
    def __init__(self, xml=None):
        # Test if broadband calibration values exists, if not use nominal values and extrapolate
        if xml['frequencies']:
            print("Broadband calibration values exists")
            self.frequencies = xml['frequencies']
            self.gain = xml['gain']
            self.angle_offset_athwartship = xml['angle_offset_alongship']
            self.angle_offset_alongship = xml['angle_offset_alongship']
            self.beam_width_athwartship = xml['beam_width_athwartship']
            self.beam_width_alongship = xml['beam_width_alongship']
            self.isCalibrated = True
        elif not xml['frequencies']:
            print("Broadband calibration values does not exist - use nominal and fit function")
            self.frequencies = None
            self.gain = None
            self.angle_offset_athwartship = None
            self.angle_offset_alongship = None
            self.beam_width_athwartship = None
            self.beam_width_alongship = None
            self.isCalibrated = False
        # Calibration data (copied from EK80CalculationPaper)
        if self.frequencies is not None:
            self.frequencies = np.array(self.frequencies)
        else:
            # If no calibration make a frequency vector
            # This is used to calculate correct frequencies after signal
            # decimation

            self.frequencies = np.linspace(self.f0, self.f1, self.n_f_points)


class Filters:
    def __init__(self, xml=None):
        self.filter_v = None
        if 'FIL1' in xml and 'NaN' not in xml['FIL1']:
            self.filter_v = []
            for v in xml['FIL1'].values():
                c = v['coefficients']
                h_fl_i = np.array(c['real']) + np.array(c['imag']) * 1j
                D = v['decimationFactor']
                N_i = v['noOfCoefficients']
                self.filter_v.append({"h_fl_i": h_fl_i, "D": D, "N_i": N_i})
            self.N_v = len(self.filter_v)

class Raw3:
    def __init__(self, xml=None):
        self.offset = xml['offset']
        self.sampleCount = xml['sampleCount']
        self.y_rx_nu = None
        if 'quadrant_signals' in xml and 'NaN' not in xml['quadrant_signals']:
            self.y_rx_nu = []
            self.N_u = len(xml['quadrant_signals'].values())
            for v in xml['quadrant_signals'].values():
                self.y_rx_nu.append(np.array(v['real']) + np.array(v['imag']) * 1j)

            self.y_rx_nu = np.array(self.y_rx_nu)


class EK80DataContainer:

    # Should we use pyecholab to get the data from .raw files?
    def __init__(self, jsonfname=None):

        # Constants
        self.cont = Constants(z_td_e=75,f_s=1.5e6,n_f_points=1000)

        self.hasData = False
        if jsonfname is not None:

            self.jdict = None
            with open(jsonfname,'r') as fp:
                self.jdict = json.load(fp)

            self.trcv = Transceiver(self.jdict['XML0']['Transceiver'])
            self.parm = Parameter(self.jdict['XML0']['Parameter'])
            self.trdu = Transducer(self.jdict['XML0']['Transducer'])
            self.envr = Environment(self.jdict['XML0']['Environment'])
            self.frqp = FrequencyPar(self.jdict['XML0']['FrequencyPar'])
            self.filt = Filters(self.jdict)
            self.raw3 = Raw3(self.jdict['RAW3'])
            self.deriv = Derived(self.frqp, self.parm, self.trdu)

            self.isCalibrated = self.frqp.isCalibrated

            #self.parseEK80JSON(self.jdict)

    """
    def parseEK80JSON(self, jdict) :
        xml0 = jdict['XML0']
        self.z_rx_e = xml0['Transceiver']['z_rx_e']
        self.f0 = xml0['Parameter']['FrequencyStart']
        self.fnom = xml0['Transducer']['Frequency']  # nominal design frequency for the transducer
        self.f1 = xml0['Parameter']['FrequencyEnd']
        self.f_c = (self.f0 + self.f1) / 2.0
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
        # Calibration data (copied from EK80CalculationPaper)
        if self.frequencies is not None:
            self.frequencies = np.array(self.frequencies)
        else:
            # If no calibration make a frequency vector
            # This is used to calculate correct frequencies after signal
            # decimation

            self.frequencies = np.linspace(self.f0, self.f1, self.n_f_points)
            
        self.filter_v = None
        if 'FIL1' in jdict and 'NaN' not in jdict['FIL1']:
            self.filter_v = []
            for v in jdict['FIL1'].values():
                c = v['coefficients']
                h_fl_i = np.array(c['real']) + np.array(c['imag']) * 1j
                D = v['decimationFactor']
                N_i = v['noOfCoefficients']
                self.filter_v.append({"h_fl_i": h_fl_i, "D": D, "N_i": N_i})
            self.N_v = len(self.filter_v)
            
        raw3 = jdict['RAW3']
        self.offset = raw3['offset']
        self.sampleCount = raw3['sampleCount']
        self.y_rx_nu = None
        if 'quadrant_signals' in raw3 and 'NaN' not in raw3['quadrant_signals']:
            self.y_rx_nu = []
            self.N_u = len(raw3['quadrant_signals'].values())
            for v in raw3['quadrant_signals'].values():
                self.y_rx_nu.append(np.array(v['real']) + np.array(v['imag']) * 1j)

            self.y_rx_nu = np.array(self.y_rx_nu)
    """
