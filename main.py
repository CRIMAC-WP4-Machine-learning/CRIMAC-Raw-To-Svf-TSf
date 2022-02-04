import matplotlib.pyplot as plt
import numpy as np
from Core.EK80CalculationPaper import EK80CalculationPaper
from Core.EK80DataContainer import EK80DataContainer

# Test data
data = EK80DataContainer('./data/pyEcholabEK80data.json')

#
# Chapter IIB: Signal generation
#

# Generate ideal send pulse
y_tx_n, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    data.f0, data.f1, data.tau, data.f_s, data.slope)
y_tx_n05slope, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    data.f0, data.f1, data.tau, data.f_s, .5)

plt.figure()
plt.plot(t*1000, y_tx_n, t*1000, y_tx_n05slope)
plt.title(
    'Ideal enveloped send pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
    .format(data.f0/1000, data.f1/1000, data.slope))
plt.xlabel('time (ms)')
plt.ylabel('amplitude')
plt.savefig('./Paper/Fig_ytx.png')

#
# Chapter IIC: Signal reception
#

# The digitized received signal 'y_rx_org' is filtered and decimated
# through the filter bank and the final signal is stored in 'y_rx_nu'.
# Note that the raw files from the EK80 implementation does not store
# the initial or the intermediate steps in the raw file, but the filtered and
# decimated result only.

# In this implementation there are two filters (N_v=2). The first
# filter operates on the raw data [y_rx(N,u,0)=y_rx_org].

# The filter coefficients 'h_fl_iv' are accessible through 'data.filter_v':
data.filter_v[0]["h_fl_i"]
data.filter_v[1]["h_fl_i"]

# the corresponding decimation factors are available through
data.filter_v[0]["D"]
data.filter_v[1]["D"]

# Comment: This should be returned by the filter_v functions:
# and the corresponding samplings frequenies
f_s = data.f_s  # The initial sampling frequency
f_s0 = f_s / data.filter_v[0]["D"]
f_s1 = f_s / (
    data.filter_v[0]["D"]*data.filter_v[1]["D"])
# The final sampling freq is stored in f_s_dec
f_s_dec = f_s1

# The frequency response function of the filter is given by its
# discrete time fourier transform:
H0 = np.fft.fft(data.filter_v[0]["h_fl_i"])
H1 = np.fft.fft(data.filter_v[1]["h_fl_i"])

# Plot of the frequency response of the filters (power) (in dB)
F0 = np.arange(len(H0))*f_s/(len(H0))
F1 = np.arange(len(H1))*f_s0/(len(H1))
G0 = 20 * np.log10(np.abs(H0))
# Repeat pattern for the second filter (4 times)
F1l = np.append(F1, F1+f_s0)
F1l = np.append(F1l, F1+2*f_s0)
F1l = np.append(F1l, F1+3*f_s0)
G1 = 20 * np.log10(np.abs(H1))
G1l = np.append(G1, G1)
G1l = np.append(G1l, G1)
G1l = np.append(G1l, G1)

plt.figure()
plt.plot(F0, G0,
         F1l, G1l,
         [data.f0, data.f1], [-140, -140])
plt.xlabel('frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.xlim([0, 100000])
plt.savefig('./Paper/Fig_fir.png')

#
# Chapter IID: Pulse compression
#

# The normalized ideal transmit signal
y_tilde_tx_n = EK80CalculationPaper.calcNormalizedTransmitSignal(y_tx_n)

# Passing the normalized and ideal transmit signal through the filter bank
y_tilde_tx_nv = EK80CalculationPaper.calcFilteredAndDecimatedSignal(
    y_tilde_tx_n, data.filter_v)

# Use the normalized, filtered and decimated transmit signal from the last
# filter stage for the matched filter.
y_mf_n = y_tilde_tx_nv[-1]

plt.figure()
plt.plot(np.abs(y_mf_n))
plt.title('The absolute value of the filtered and decimated output signal')
plt.xlabel('samples ()')
plt.ylabel('amplitude')
plt.savefig('./Paper/Fig_y_mf_n.png')

# The autocorrelation function and efficient pulse duration of the mathced filter
y_mf_auto_n, tau_eff = EK80CalculationPaper.calcAutoCorrelation(
    y_mf_n, f_s_dec)

plt.figure()
plt.plot(np.abs(y_mf_auto_n))
plt.title('The autocorrelation function of the matched filter.')
plt.xlabel('samples')
plt.ylabel('ACF')
plt.savefig('./Paper/Fig_ACF.png')

# Calculating the pulse compressed quadrant signals separately on each channel
y_pc_nu = EK80CalculationPaper.calcPulseCompressedQuadrants(
    data.y_rx_nu,
    y_mf_n)

# Calculating the average signal over the channels
y_pc_n = EK80CalculationPaper.calcAvgSumQuad(y_pc_nu)

# Calculating the average signal over paired fore, aft, starboard, port channel
y_pc_halves_n = EK80CalculationPaper.calcTransducerHalves(
    y_pc_nu)

#
# Chapter IIE: Power angles and samples
#

# Calcuate the power across transducer channels
p_rx_e_n = EK80CalculationPaper.calcPower(
    y_pc_n,
    data.z_td_e,
    data.z_rx_e,
    data.N_u)

# Calculate the angle sensitivities
gamma_theta = EK80CalculationPaper.calcGamma(
    data.angle_sensitivity_alongship_fnom,
    data.f_c,
    data.fnom)
gamma_phi = EK80CalculationPaper.calcGamma(
    data.angle_sensitivity_athwartship_fnom,
    data.f_c,
    data.fnom)

# Calculate the physical angles
y_theta_n, y_phi_n = EK80CalculationPaper.calcAngles(
    y_pc_halves_n,
    gamma_theta,
    gamma_phi)

# Plot angles
plt.figure()
plt.plot(y_theta_n)
plt.plot(y_phi_n)
plt.title('The physical angles.')
plt.xlabel(' ')
plt.ylabel('angles')
plt.savefig('./Paper/Fig_theta_phi.png')

#
# Chapter III: TARGET STRENGTH
#

#Sp_n, r_n = EK80CalculationPaper.calcSp(p_rx_e, args.r0, args.r1)


#
# Chapter IV: VOLUME BACKSCATTERING STRENGTH
#

# Calculate Sv
# Sv =






