import matplotlib.pyplot as plt
import numpy as np
from Core.EK80CalculationPaper import EK80CalculationPaper
from Core.EK80DataContainer import EK80DataContainer

# Test data
data = EK80DataContainer('./data/pyEcholabEK80data.json')


"""
cont = data.cont
trcv = data.trcv
parm = data.parm
trdu = data.trdu
envr = data.envr
frqp = data.frqp
filt = data.filt
raw3 = data.raw3
deriv = data.deriv
"""
# data.trdu.f_c
# data.f_c -> I paper: $f_c$

# To be considered:
# f0 -> f_0
# f1 -> f_1

# Unpack variabels
z_td_e, f_s, n_f_points = data.cont.getParameters()

z_rx_e = data.trcv.getParameters()

f0, f1, f_c, tau, slope, sampleInterval, p_tx_e = data.parm.getParameters()

fnom, G_fnom, PSI_fnom, angle_offset_alongship_fnom, \
angle_offset_athwartship_fnom, angle_sensitivity_alongship_fnom, \
angle_sensitivity_athwartship_fnom, beam_width_alongship_fnom, \
beam_width_alongship_fnom, corrSa = data.trdu.getParameters()

c, alpha, temperature, salinity, \
acidity, latitude, depth, dropKeelOffset = data.envr.getParameters()
frequencies, gain, angle_offset_athwartship, angle_offset_alongship, \
beam_width_athwartship, beam_width_alongship = data.frqp.getParameters()

filter_v, N_v = data.filt.getParameters()

offset, sampleCount, y_rx_nu, N_u, y_rx_nu = data.raw3.getParameters()

g_0_f_c, lambda_f_c, PSI_f = data.deriv.getParameters()

#
# Chapter IIB: Signal generation
#

# Generate ideal send pulse
y_tx_n, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    f0, f1, tau, f_s, slope)

y_tx_n05slope, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    f0, f1, tau, f_s, .5)

plt.figure()
plt.plot(t*1000, y_tx_n, t*1000, y_tx_n05slope)
plt.title(
    'Ideal enveloped send pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
    .format(f0/1000, f1/1000, slope))
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
filter_v[0]["h_fl_i"]
filter_v[1]["h_fl_i"]

# the corresponding decimation factors are available through
filter_v[0]["D"]
filter_v[1]["D"]

# The sampling freq for each filter step
f_s_dec_v = EK80CalculationPaper.calcDecmiatedSamplingRate(filter_v, f_s)
# The final sampling frequency
f_s_dec = f_s_dec_v[-1]

# The frequency response function of the filter is given by its
# discrete time fourier transform:
H0 = np.fft.fft(filter_v[0]["h_fl_i"])
H1 = np.fft.fft(filter_v[1]["h_fl_i"])

# Plot of the frequency response of the filters (power) (in dB)
F0 = np.arange(len(H0))*f_s_dec_v[0]/(len(H0))
F1 = np.arange(len(H1))*f_s_dec_v[1]/(len(H1))
G0 = 20 * np.log10(np.abs(H0))
# Repeat pattern for the second filter (4 times)
F1l = np.append(F1, F1+f_s_dec_v[1])
F1l = np.append(F1l, F1+2*f_s_dec_v[1])
F1l = np.append(F1l, F1+3*f_s_dec_v[1])
G1 = 20 * np.log10(np.abs(H1))
G1l = np.append(G1, G1)
G1l = np.append(G1l, G1)
G1l = np.append(G1l, G1)

plt.figure()
plt.plot(F0, G0,
         F1l, G1l,
         [f0, f1], [-140, -140])
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
    y_tilde_tx_n, filter_v)

# Use the normalized, filtered and decimated transmit signal from the last
# filter stage for the matched filter.
y_mf_n = y_tilde_tx_nv[-1]

plt.figure()
plt.plot(np.abs(y_mf_n))
plt.title('The absolute value of the filtered and decimated output signal')
plt.xlabel('samples ()')
plt.ylabel('amplitude')
plt.savefig('./Paper/Fig_y_mf_n.png')

# The autocorrelation function and efficient pulse duration of the mathced
# filter
y_mf_auto_n, tau_eff = EK80CalculationPaper.calcAutoCorrelation(
    y_mf_n, f_s_dec)

plt.figure()
plt.plot(np.abs(y_mf_auto_n))
plt.title('The autocorrelation function of the matched filter.')
plt.xlabel('samples')
plt.ylabel('ACF')
plt.savefig('./Paper/Fig_ACF.png')

# Calculating the pulse compressed quadrant signals separately on each channel
y_pc_nu = EK80CalculationPaper.calcPulseCompressedQuadrants(y_rx_nu, y_mf_n)

# Calculating the average signal over the channels
y_pc_n = EK80CalculationPaper.calcAvgSumQuad(y_pc_nu)

# Calculating the average signal over paired fore, aft, starboard, port channel
y_pc_halves_n = EK80CalculationPaper.calcTransducerHalves(y_pc_nu)

#
# Chapter IIE: Power angles and samples
#

# Calcuate the power across transducer channels
p_rx_e_n = EK80CalculationPaper.calcPower(
    y_pc_n,
    z_td_e,
    z_rx_e,
    N_u)

# Calculate the angle sensitivities
gamma_theta = EK80CalculationPaper.calcGamma(
    angle_sensitivity_alongship_fnom,
    f_c,
    fnom)
gamma_phi = EK80CalculationPaper.calcGamma(
    angle_sensitivity_athwartship_fnom,
    f_c,
    fnom)

# Calculate the physical angles
theta_n, phi_n = EK80CalculationPaper.calcAngles(
    y_pc_halves_n,
    gamma_theta,
    gamma_phi)

# Plot angles
plt.figure()
plt.plot(theta_n)
plt.plot(phi_n)
plt.title('The physical angles.')
plt.xlabel(' ')
plt.ylabel('angles')
plt.savefig('./Paper/Fig_theta_phi.png')

#
# Chapter III: TARGET STRENGTH
#

# Get parameters
r0 = 10
r1 = 30
before = 0.5
after = 1

#p_tx_e = parm.ptx
#f_c = parm.f_c
#g_0_f_c = data.deriv.g_0_f_c
#PSI_f = data.deriv.PSI_f
#lambda_f_c = data.deriv.lambda_f_c

# logSpCf - > lambda_f_c
# power -> p_tx_e
# Gfc -> g_0_f_c
# r -> r_n
# Sp -> S_p_n

#
# <RUBEN!
#


# Will be written explicitly in EK80CalculationPaper and removed
# logSpCf = EK80CalculationPaper.calculateCSpfdB(f_c, ptx)
# Move to EK80DataContainer (Ruben)
r_n, _ = EK80CalculationPaper.calcRange(
    sampleInterval,
    sampleCount,
    c,
    offset)

# alpha_fc = self.calcAbsorption(self.temperature, self.salinity,
# self.depth, self.acidity, self.c, self.f_c)

# Move to EK80DataContainer (Ruben):
alpha_f_c = EK80CalculationPaper.calcAbsorption(
    temperature,
    salinity,
    depth,
    acidity,
    c,
    f_c)

# Move to data container (Ruben)
# G = np.interp1(X, g_theta_phi_f, Y)

#
# RUBEN>
#


# Calculate the point scattering strength (Sp)
Sp_n = EK80CalculationPaper.calcSp(
    p_rx_e_n,
    r_n,
    alpha_f_c,
    p_tx_e,
    lambda_f_c,
    g_0_f_c,
    r0,
    r1)

# Data from a single target (_t denotes data from single target)
r_t, theta_t, phi_t, y_pc_t_n, dum_p, dum_theta, dum_phi = \
    EK80CalculationPaper.singleTarget(
        y_pc_n, p_rx_e_n, theta_n, phi_n, r_n,
        r0, r1, before, after)


# Pick the reduced samples from the mathed filtered decimated send pulse
y_mf_auto_red_n = EK80CalculationPaper.alignAuto(y_mf_auto_n, y_pc_t_n)
# NB: In the example these are the same. Perhaps chose differently for
# illustrative purposes?

# DFT on the pulse compressed received signal and the pulse compressed
# send pulse signal (reduced and matched filtered)

Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m, f_m_t = \
    EK80CalculationPaper.calcDFTforTS(y_pc_t_n, y_mf_auto_red_n,
                                      n_f_points, f0, f1, f_s_dec)

fig, axs = plt.subplots(3)
fig.suptitle('Normalized DFT')
axs[0].plot(f_m_t, np.abs(Y_pc_t_m))
axs[0].set_ylabel('Y_tilde_pc_t_m')
axs[1].plot(f_m_t, np.abs(Y_mf_auto_red_m))
axs[1].set_ylabel('Y_mf_auto_red_m')
axs[2].plot(f_m_t, np.abs(Y_tilde_pc_t_m))
axs[2].set_xlabel('f (Hz)')
axs[2].set_ylabel('Y_tilde_pc_t_m')
plt.savefig('./Paper/Fig_normalizedDFT.png')


# Calculate the power by frequency from a single target
P_rx_e_t_m = EK80CalculationPaper.calcPowerFreq(
    N_u,
    Y_tilde_pc_t_m,
    z_td_e,
    z_rx_e)

# Calculate the target strength
g_theta_phi_f = 1
#TS_m = EK80CalculationPaper.calcTSf(
#    P_rx_e_t_m, r_t, alpha, p_tx_e, lambda_f_c,
#    g_theta_phi_f, theta_t, phi_t, f_m_t)


#
# Chapter IV: VOLUME BACKSCATTERING STRENGTH
#

# Calculate Sv
# Sv =

