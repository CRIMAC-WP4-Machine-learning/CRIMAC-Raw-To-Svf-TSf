import matplotlib.pyplot as plt
import numpy as np
from Core.EK80CalculationPaper import EK80CalculationPaper
from Core.EK80DataContainer import EK80DataContainer

# Test data
data = EK80DataContainer('./data/pyEcholabEK80data.json')

cont = data.cont
trcv = data.trcv
parm = data.parm
trdu = data.trdu
envr = data.envr
frqp = data.frqp
filt = data.filt
raw3 = data.raw3
#
# Chapter IIB: Signal generation
#

# Generate ideal send pulse
y_tx_n, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    parm.f0,
    parm.f1,
    parm.tau,
    cont.f_s,
    parm.slope)

y_tx_n05slope, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
    parm.f0,
    parm.f1,
    parm.tau,
    cont.f_s, .5)

plt.figure()
plt.plot(t*1000, y_tx_n, t*1000, y_tx_n05slope)
plt.title(
    'Ideal enveloped send pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
    .format(parm.f0/1000, parm.f1/1000, parm.slope))
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
filt.filter_v[0]["h_fl_i"]
filt.filter_v[1]["h_fl_i"]

# the corresponding decimation factors are available through
filt.filter_v[0]["D"]
filt.filter_v[1]["D"]

# The sampling freq for each filter step
f_s_dec_v = EK80CalculationPaper.calcDecmiatedSamplingRate(
    filt.filter_v, cont.f_s)
# The final sampling frequency
f_s_dec = f_s_dec_v[-1]

# The frequency response function of the filter is given by its
# discrete time fourier transform:
H0 = np.fft.fft(filt.filter_v[0]["h_fl_i"])
H1 = np.fft.fft(filt.filter_v[1]["h_fl_i"])

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
         [parm.f0, parm.f1], [-140, -140])
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
    y_tilde_tx_n, filt.filter_v)

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
    raw3.y_rx_nu,
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
    cont.z_td_e,
    trcv.z_rx_e,
    raw3.N_u)

# Calculate the angle sensitivities
gamma_theta = EK80CalculationPaper.calcGamma(
    trdu.angle_sensitivity_alongship_fnom,
    parm.f_c,
    trdu.fnom)
gamma_phi = EK80CalculationPaper.calcGamma(
    trdu.angle_sensitivity_athwartship_fnom,
    parm.f_c,
    trdu.fnom)

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
"""
# I have copied in most of the parameters used by "self" as candidates to EK80DataContainer:

# Sp:
# Gfc = self.calc_G0_m(self.f_c)
# PSIfc = self.PSI_f(self.f_c)
# logSpCf = self.calculateCSpfdB(self.f_c)
# r, _ = self.calcRange()
# alpha_fc = self.calcAbsorption(self.temperature, self.salinity, self.depth, self.acidity, self.c, self.f_c)

# singleTarget:
# r_n, _ = self.calcRange()

# CalcTSf:
# L = self.n_f_points
# y_mf_auto_red_n = self.alignAuto(self.y_mf_auto_n, y_pc_t_n)
# Y_pc_t_m = self.freqtransf(_Y_pc_t_m, self.f_s_dec, f_m)
# Y_mf_auto_red_m = self.freqtransf(_Y_mf_auto_red_m, self.f_s_dec, f_m)
# G0_m = self.calc_G0_m(f_m)
# B_theta_phi_m = self.calc_B_theta_phi_m(theta, phi, f_m)
# G_theta_phi_m = G0_m - B_theta_phi_m
# alpha_m = self.calcAbsorption(self.temperature, self.salinity, self.depth, self.acidity, self.c, f_m)
# logSpCf = self.calculateCSpfdB(f_m)
# Y_tilde_pc_t_m = Y_pc_t_m / Y_mf_auto_red_m
# P_rx_e_t_m = self.C1Prx * np.abs(Y_tilde_pc_t_m) ** 2


# Sp
r0 = 10
r1 = 30
before = 0.5
after = 1

# Calculate the point scattering strength (Sp)
Sp_n, r_n = EK80CalculationPaper.calcSp(p_rx_e_n, r0, r1)

# Extract single target
r, theta, phi, y_pc_t_n = EK80CalculationPaper.singleTarget(
    y_pc_n, p_rx_e_n, theta_n, phi_n,
    r0, r1, before, after)

# Calculate the target strength of the single target
TS_m, f_m,  = EK80CalculationPaper.calcTSf(r, theta, phi, y_pc_t_n)

#
# Chapter IV: VOLUME BACKSCATTERING STRENGTH
#

# Calculate Sv
# Sv =


"""
