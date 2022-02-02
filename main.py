import matplotlib.pyplot as plt
import numpy as np
from Core.EK80Calculation import EK80Calculation
# from Core.EK80DataContainer import EK80DataContainer

# Test data

# The data file ./data/pyEcholabEK80data.json contain data from one
# ping from the EK80 echosounder, including the

ekcalc = EK80Calculation('./data/pyEcholabEK80data.json')

#
# Chapter: Signal generation
#

# Generate ideal send pulse
f0 = ekcalc.f0  # frequency (Hz)
f1 = ekcalc.f1  # frequency (Hz)
tau = ekcalc.tau  # time (s)
fs = ekcalc.f_s  # frequency (Hz)
slope = ekcalc.slope
y_tx_n, t = EK80Calculation.generateIdealWindowedSendPulse(
    f0, f1, tau, fs, slope)
y_tx_n05slope, t = EK80Calculation.generateIdealWindowedSendPulse(
    f0, f1, tau, fs, .5)

plt.figure()
plt.plot(t*1000, y_tx_n, t*1000, y_tx_n05slope)
plt.title(
    'Ideal enveloped send pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
    .format(f0/1000, f1/1000, slope))
plt.xlabel('time (ms)')
plt.ylabel('amplitude')
plt.savefig('Fig_ytx.png')

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
# The filter coefficients 'h_fl_iv' are accessible through the 'fil1s' variable:
ekcalc.fil1s[0].Coefficients
ekcalc.fil1s[1].Coefficients
# and the corresponding decimation factor is available through
ekcalc.fil1s[0].DecimationFactor
ekcalc.fil1s[1].DecimationFactor

# [Note from Nils Olav: I think that 'fil1s' should be renamed 'h_fl_iv' to
# conform with the paper.]

# The frequency response function of the filter is given by its
# discrete time fourier transform:
H0 = np.fft.fft(ekcalc.fil1s[0].Coefficients)
H1 = np.fft.fft(ekcalc.fil1s[1].Coefficients)

# Plot of the frequency response of the filters (power) (in dB)
# [Note from Nils Olav: someone needs to review this, I am not an
# expert here and I am not sure hwo this works for complex filters
# and how to deal with the higher freqs (that are symm and typically
# removed for real input)]
plt.figure()
#plt.plot(np.fft.fftfreq(len(H0))*fs/1000, 20 * np.log10(H0/len(H0)),'.',
#         np.fft.fftfreq(len(H1))*fs/1000, 20 * np.log10(H1/len(H1)),'.')
plt.semilogx(np.arange(len(H0)), 20 * np.log10(np.abs(H0)),
             np.arange(len(H1)), 20 * np.log10(np.abs(H1)))
plt.xlabel('Normalized Frequency')
plt.ylabel('Gain [dB]')
plt.savefig('Fig_fir.png')

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
plt.savefig('Fig_y_mf_n.png')

# The autocorrelation function of the matched filter signal
# ekcalc.y_mf_auto

plt.figure()
plt.plot(np.abs(ekcalc.y_mf_auto))
plt.title('The autocorrelation function of the matched filter.')
plt.xlabel('samples')
plt.ylabel('ACF')
plt.savefig('Fig_ACF.png')


# Calculating the pulse compressed quadrant signals separately on each channel
y_pc_nu = ekcalc.calcPulseCompressedQuadrants(ekcalc.y_rx_nu)

# Calculating the average signal over the channels
y_pc_n = ekcalc.calcAvgSumQuad(y_pc_nu)

# Calculating the average signal over paired channels

# This is done inside the calcElectricalAngles(self, y_pc) function and
# the results are not directly accessible.

#print('The four quadrants and the length of the sampled data: ' +
#      str(np.shape(ekcalc.y_rx_org)))

#
# Chapter: Power angles and samples
#

#p_rx_e = ekcalc.calcPower(y_pc_n)

#y_theta_n, y_phi_n = ekcalc.calcElectricalAngles(y_pc_nu)


