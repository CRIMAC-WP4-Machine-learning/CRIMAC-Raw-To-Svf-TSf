import numpy as np
import json

from Core.FIL1 import FIL1

# Ruben: Create a logic that separate derived variables and .raw variables.
# required changes:
# logSpCf - > lambda_f_c
# power -> p_tx_e
# Gfc -> g_0_f_c
# r -> r_n
# Sp -> S_p_n


class Derived:
    def __init__(self, frqp, parm, trdu, envr):

        self.Gfc = self.calc_Gfc(frqp, parm,trdu)
        self.PSI_f = trdu.PSI_fnom + 20 * np.log10(trdu.fnom/frqp.frequencies)
        self.g_0_f_c = np.power(self.Gfc/10, 10)
        self.lambda_f_c = envr.c/parm.f_c

    @staticmethod
    def calc_Gfc(frqp, parm, trdu):
        if frqp.isCalibrated:
            # Calibrated case
            return np.interp(parm.f_c, frqp.frequencies, frqp.gain)
        else:
            # Uncalibrated case
            return trdu.G_fnom + 20 * np.log10(frqp.frequencies / trdu.fnom)


    def getParameters(self):
        return self.g_0_f_c, self.lambda_f_c, self.PSI_f


class Constants:
    def __init__(self, z_td_e, f_s, n_f_points):
        self.z_td_e = z_td_e
        self.f_s = f_s
        self.n_f_points = n_f_points

    def getParameters(self):
        return self.z_td_e, self.f_s, self.n_f_points


class Transceiver:
    def __init__(self, xml):
        self.z_rx_e = xml['z_rx_e']

    def getParameters(self):
        return self.z_rx_e


class Parameter:
    def __init__(self, xml):
        self.f0 = xml['FrequencyStart']
        self.f1 = xml['FrequencyEnd']
        self.f_c = (self.f0 + self.f1) / 2.0
        self.tau = xml['PulseDuration']
        self.slope = xml['Slope']
        self.sampleInterval = xml['SampleInterval']
        self.p_tx_e = xml['TransmitPower']

    def getParameters(self):
        return self.f0, self.f1, self.f_c, self.tau, self.slope, self.sampleInterval, self.p_tx_e

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

    def getParameters(self):
        return self.fnom, self.G_fnom, self.PSI_fnom, self.angle_offset_alongship_fnom, \
               self.angle_offset_athwartship_fnom,self.angle_sensitivity_alongship_fnom, \
               self.angle_sensitivity_athwartship_fnom, self.beam_width_alongship_fnom, \
               self.beam_width_athwartship_fnom, self.corrSa

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

    def getParameters(self):
        return self.c, self.alpha, self.temperature, self.salinity, \
               self.acidity, self.latitude,self.depth, self.dropKeelOffset

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

    def getParameters(self):
        return self.frequencies, self.gain, self.angle_offset_athwartship, self.angle_offset_alongship, \
               self.beam_width_athwartship, self.beam_width_alongship


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

    def getParameters(self):
        return self.filter_v, self.N_v

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

    def getParameters(self):
        return self.offset, self.sampleCount, self.y_rx_nu, self.N_u, self.y_rx_nu

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
            self.deriv = Derived(self.frqp, self.parm, self.trdu, self.envr)

            self.isCalibrated = self.frqp.isCalibrated

    def calc_angle_offsets_m(self, f):
        if self.isCalibrated:
            # Calibrated case
            angle_offset_alongship_m = np.interp(f, self.frqp.frequencies, self.frqp.angle_offset_alongship)
            angle_offset_athwartship_m = np.interp(f, self.frqp.frequencies, self.frqp.angle_offset_athwartship)
        else:
            # Uncalibrated case
            angle_offset_alongship_m = self.trdu.angle_offset_alongship_fnom * np.ones(len(f))
            angle_offset_athwartship_m = self.trdu.angle_offset_athwartship_fnom * np.ones(len(f))
        return angle_offset_alongship_m, angle_offset_athwartship_m

    def calc_beam_widths_m(self, f):
        if self.isCalibrated:
            # Calibrated case
            beam_width_alongship_m = np.interp(f, self.frqp.frequencies, self.frqp.beam_width_alongship)
            beam_width_athwartship_m = np.interp(f, self.frqp.frequencies, self.frqp.beam_width_athwartship)
        else:
            # Uncalibrated case
            beam_width_alongship_m = self.trdu.beam_width_alongship_fnom * self.trdu.fnom / f
            beam_width_athwartship_m = self.trdu.beam_width_athwartship_fnom * self.trdu.fnom / f
        return beam_width_alongship_m, beam_width_athwartship_m

    def calc_B_theta_phi_m(self, theta, phi, f):
        angle_offset_alongship_m, angle_offset_athwartship_m = self.calc_angle_offsets_m(f)
        beam_width_alongship_m, beam_width_athwartship_m = self.calc_beam_widths_m(f)

        B_theta_phi_m = 0.5 * 6.0206 * ((np.abs(theta - angle_offset_alongship_m) / (beam_width_alongship_m / 2)) ** 2 + \
                                        (np.abs(phi - angle_offset_athwartship_m) / (
                                                    beam_width_athwartship_m / 2)) ** 2 - \
                                        0.18 * ((np.abs(theta - angle_offset_alongship_m) / (
                            beam_width_alongship_m / 2)) ** 2 * \
                                                (np.abs(phi - angle_offset_athwartship_m) / (
                                                            beam_width_athwartship_m / 2)) ** 2))