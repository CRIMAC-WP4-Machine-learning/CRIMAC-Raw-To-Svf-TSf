import matplotlib.pyplot as plt
import numpy as np
from Core.EK80CalculationPaper import EK80CalculationPaper
from Core.EK80Calculation import EK80Calculation

# Test data

# The data file ./data/pyEcholabEK80data.json contain data from one
# ping from the EK80 echosounder, including the
ekcalc = EK80CalculationPaper('./data/pyEcholabEK80data.json')

#
# Chapter: Signal generation
#

# Generate ideal send pulse
f0 = ekcalc.f0  # frequency (Hz)
f1 = ekcalc.f1  # frequency (Hz)
tau = ekcalc.tau  # time (s)
fs = ekcalc.f_s  # frequency (Hz)
slope = ekcalc.slope
y_tx_n, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    f0, f1, tau, fs, slope)
y_tx_n05slope, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    f0, f1, tau, fs, .5)

plt.figure()
plt.plot(t*1000, y_tx_n, t*1000, y_tx_n05slope)
plt.title(
    'Ideal enveloped send pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
    .format(f0/1000, f1/1000, slope))
plt.xlabel('time (ms)')
plt.ylabel('amplitude')
plt.savefig('./Paper/Fig_ytx.png')

#
# Chapter: Signal reception
#

# The digitized received signal 'y_rx_org' is filtered and decimated
# through the filter bank and the final signal is stored in 'y_rx_nu'.
# Note that the raw files from the EK80 implementation does not store
# the initial or the intermediate steps in the raw file, but the filtered and
# deciated result only, The data is stored in the variable 'y_rx_nu'.

# In this implementation there are two filters (N_v=2). The first
# filter operates on the raw data [y_rx(N,u,0)=y_rx_org].
# The filter coefficients 'h_fl_iv' are accessible through the 'fil1s' variable
# The 'fil1s' field corresponds to content of the EK80 datagram in the EK80 raw
# file
ekcalc.fil1s[0].Coefficients
ekcalc.fil1s[1].Coefficients
# and the corresponding decimation factor is available through
ekcalc.fil1s[0].DecimationFactor
ekcalc.fil1s[1].DecimationFactor
# The corresponding samplings frequenies
f_s = ekcalc.f_s  # The initial sampling frequency
f_s0 = f_s / ekcalc.fil1s[0].DecimationFactor
f_s1 = f_s / (
    ekcalc.fil1s[0].DecimationFactor*ekcalc.fil1s[1].DecimationFactor)

# The frequency response function of the filter is given by its
# discrete time fourier transform:
H0 = np.fft.fft(ekcalc.fil1s[0].Coefficients)
H1 = np.fft.fft(ekcalc.fil1s[1].Coefficients)

# Plot of the frequency response of the filters (power) (in dB)
F0 = np.arange(len(H0))*f_s/(len(H0))
F1 = np.arange(len(H1))*f_s0/(len(H1))
G0 = 20 * np.log10(np.abs(H0))
# Repeat pattern for the second filter (3 times)
F1l = np.append(F1, F1+f_s0)
F1l = np.append(F1l, F1+2*f_s0)
G1 = 20 * np.log10(np.abs(H1))
G1l = np.append(G1, G1)
G1l = np.append(G1l, G1)

plt.figure()
plt.plot(F0, G0,
         F1l, G1l,
         [ekcalc.f0, ekcalc.f1], [-140, -140])
plt.xlabel('frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.xlim([0, 70000])
plt.savefig('./Paper/Fig_fir.png')

#
# Chapter: Pulse compression
#

# The normalized ideal transmit signal (y_tilde_tx_n) is passed through the
# filter banks to generate the matched filter.
# Run ekcalc.calculateDerivedVariables? to see the explanation of the vars

# The source signal filtered through the final decimation is stored in
# ekcalc.y_rx_org

plt.figure()
plt.plot(np.abs(ekcalc.y_mf_n))
plt.title('The absolute value of the filtered and decimated output signal')
plt.xlabel('samples ()')
plt.ylabel('amplitude')
plt.savefig('./Paper/Fig_y_mf_n.png')

# The autocorrelation function of the matched filter signal
# ekcalc.y_mf_auto

plt.figure()
plt.plot(np.abs(ekcalc.y_mf_auto_n))
plt.title('The autocorrelation function of the matched filter.')
plt.xlabel('samples')
plt.ylabel('ACF')
plt.savefig('./Paper/Fig_ACF.png')

# Calculating the pulse compressed quadrant signals separately on each channel
y_pc_nu = EK80CalculationPaper.calcPulseCompressedQuadrants(ekcalc.y_rx_nu,
                                                            ekcalc.y_mf_n)

# Calculating the average signal over the channels
y_pc_n = EK80CalculationPaper.calcAvgSumQuad(y_pc_nu)

# Calculating the average signal over paired fore, aft, starboard, port channel
y_pc_halves = EK80CalculationPaper.calc_transducer_halves(y_pc_nu)


#
# Chapter IIE: Power angles and samples
#

# Calcuate the power across transducer channels
p_rx_e = ekcalc.calcPower(y_pc_n)

# Calculate the angle sensitivities
gamma_theta = EK80CalculationPaper.calcGamma(
    ekcalc.angle_sensitivity_alongship_fnom,
    ekcalc.f_c,
    ekcalc.fnom)
gamma_phi = EK80CalculationPaper.calcGamma(
    ekcalc.angle_sensitivity_athwartship_fnom,
    ekcalc.f_c,
    ekcalc.fnom)

# Calculate the physical angles
y_theta_n, y_phi_n = EK80CalculationPaper.calcAngles(
    y_pc_halves,
    gamma_theta,
    gamma_phi)

#
# Chapter III: TARGET STRENGTH
#

# p_rx_e_n = ekcalc.calcPower(y_pc_n,c)
#    def calcPower(y_pc_n, z_rx_e, z_td_e, N_u):

#z_rx_e
#impedac.z.z_td_e

#p_rx_e_n = EK80Calculation.calcPower(y_pc_n)

#theta_n, phi_n = ekcalc.calcElectricalAngles(y_pc_nu)






