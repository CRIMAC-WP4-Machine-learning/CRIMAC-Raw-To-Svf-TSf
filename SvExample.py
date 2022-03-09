import matplotlib.pyplot as plt
import numpy as np
from Core.Calculation import Calculation
from Core.EK80DataContainer import EK80DataContainer
from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap

#
# Test data Sv - Herring school, single ping
#

data = EK80DataContainer('./data/CRIMAC_Svf.json')  # Sv school 

# Unpack variabels
z_td_e, f_s, n_f_points = data.cont.getParameters()

z_rx_e = data.trcv.getParameters()

f_0, f_1, f_c, tau, slope, sampleInterval, p_tx_e = data.parm.getParameters()

# The frequency vector for both Ts and Sv (grid for index m)
f_m = np.linspace(f_0, f_1, n_f_points)

f_n, G_fnom, Psi_f_n, angle_offset_alongship_fnom, \
    angle_offset_athwartship_fnom, angle_sensitivity_alongship_fnom, \
    angle_sensitivity_athwartship_fnom, beam_width_alongship_fnom, \
    beam_width_alongship_fnom, corrSa = data.trdu.getParameters()

c, alpha, temperature, salinity, \
    acidity, latitude, depth, dropKeelOffset = data.envr.getParameters()

frequencies, gain, angle_offset_athwartship, angle_offset_alongship, \
    beam_width_athwartship, beam_width_alongship = data.frqp.getParameters()

filter_v, N_v = data.filt.getParameters()

offset, sampleCount, y_rx_nu, N_u, y_rx_nu = data.raw3.getParameters()

g_0_f_c, lambda_f_c, _ = data.deriv.getParameters()

r_n, dr = data.calcRange(
    sampleInterval,
    sampleCount,
    c,
    offset)

alpha_f_c = data.calcAbsorption(
    temperature,
    salinity,
    depth,
    acidity,
    c,
    f_c)

# The on axix gain as a function of f_m
g_0_m = data.calc_g(0, 0, f_m)

# Cacluate lambda and alpha on the f_m grid
lambda_m = data.calc_lambda_f(f_m)
alpha_m = data.calc_alpha_f(f_m)

# Calculate Psi for f_c and on the f_m grid
Psi_f_c = Calculation.calc_Psi_f(Psi_f_n, f_n, f_c)
Psi_m = Calculation.calc_Psi_f(Psi_f_n, f_n, f_m)

#
# Precalculations equal to TSExample.py
#

# Generate ideal send pulse
y_tx_n, t = Calculation.generateIdealWindowedSendPulse(
    f_0, f_1, tau, f_s, slope)

# The sampling freq for each filter step
f_s_dec_v = Calculation.calcDecmiatedSamplingRate(filter_v, f_s)
# The final sampling frequency
f_s_dec = f_s_dec_v[-1]

# The normalized ideal transmit signal
y_tilde_tx_n = Calculation.calcNormalizedTransmitSignal(y_tx_n)

# Passing the normalized and ideal transmit signal through the filter bank
y_tilde_tx_nv = Calculation.calcFilteredAndDecimatedSignal(
    y_tilde_tx_n, filter_v)

# Use the normalized, filtered and decimated transmit signal from the last
# filter stage for the matched filter.
y_mf_n = y_tilde_tx_nv[-1]

# The autocorrelation function and efficient pulse duration of the mathced
# filter
y_mf_auto_n, tau_eff = Calculation.calcAutoCorrelation(
    y_mf_n, f_s_dec)

# Calculating the pulse compressed quadrant signals separately on each channel
y_pc_nu = Calculation.calcPulseCompressedQuadrants(y_rx_nu, y_mf_n)

# Calculating the average signal over the channels
y_pc_n = Calculation.calcAvgSumQuad(y_pc_nu)

# Calculating the average signal over paired fore, aft, starboard, port channel
y_pc_halves_n = Calculation.calcTransducerHalves(y_pc_nu)

# Calcuate the power across transducer channels
p_rx_e_n = Calculation.calcPower(
    y_pc_n,
    z_td_e,
    z_rx_e,
    N_u)

# Calculate the angle sensitivities
gamma_theta = Calculation.calcGamma(
    angle_sensitivity_alongship_fnom,
    f_c,
    f_n)
gamma_phi = Calculation.calcGamma(
    angle_sensitivity_athwartship_fnom,
    f_c,
    f_n)

# Calculate the physical angles
theta_n, phi_n = Calculation.calcAngles(
    y_pc_halves_n,
    gamma_theta,
    gamma_phi)

#
# Chapter IV: VOLUME BACKSCATTERING STRENGTH
#


# Calculate average Sv
# TODO: I get zero power in the p_rx_e_n. Fails when doing log10. "Quickfix":
p_rx_e_n = p_rx_e_n + .0000000000000001

# TODO: Range equal to zero will not work. either remove first sample or
# reconsider the range vector (log10(0) does not exist). Hack:
r_n[r_n == 0] = 0.0000000001
Sv_n = Calculation.calc_Sv(p_rx_e_n, r_n, lambda_f_c,
                                    p_tx_e, alpha_f_c, c, tau_eff,
                                    Psi_f_c, g_0_f_c)

# Calculate the pulse compressed signal adjusted for spherical loss
y_pc_s_n = Calculation.calc_PulseCompSphericalSpread(y_pc_n, r_n)

# Hanning window
w_tilde_i, N_w, t_w, t_w_n = Calculation.defHanningWindow(c, tau, dr,
                                                                   f_s_dec)

# Calculate the DFT on the pulse compressed signal

step = 1  # Needs some thoughts... 50% overlapp

# Sjekk at n_f_ponts er 2 pulslengder, det er det vi vil ha
Y_pc_v_m_n, Y_mf_auto_m, Y_tilde_pc_v_m_n, svf_range \
    = Calculation.calcDFTforSv(
        y_pc_s_n, w_tilde_i, y_mf_auto_n, N_w, n_f_points, f_m, f_s_dec,
        r_n, step)

# Calculate the power
P_rx_e_t_m_n = Calculation.calcPowerFreqforSv(
    Y_tilde_pc_v_m_n, N_u, z_rx_e, z_td_e)

# Calculate the Sv(f)
# TODO: Range == 0 does not work well with log10. another hack:

Sv_m_n = Calculation.calcSvf(P_rx_e_t_m_n,
                                      alpha_m, p_tx_e, lambda_m, t_w,
                                      Psi_m, g_0_m, c, svf_range)
        


fig, axs = plt.subplots(2)
axs[0].plot(t_w_n, w_tilde_i)
axs[0].set_ylabel('w_tilde_i')
axs[1].plot(t_w_n, w_tilde_i)
axs[1].set_ylabel('w_tilde_i')
axs[1].set_xlabel('t (s)')
plt.savefig('./Paper/Fig_Svf.png')

# Plot Sv(f) for entire ping
fig = figure()
plt.pcolormesh(f_m/1000,svf_range,Sv_m_n,vmin=-180, vmax=-120, shading='auto')
plt.colorbar()
plt.title('Echogram [Sv]')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Range [m]')
plt.show()

indices=np.where(np.logical_and(svf_range>=15, svf_range<=34))
Sv=[]

# Plot Sv(f) in one depth in the middle of layer
plt.plot(Sv_m_n[int(len(indices[0]) / 2) - 1,])
plt.title('Sv(f) at one depth')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Sv')
plt.grid()

for i in range(len(f_m)):
    sv=10**(Sv_m_n[indices,i]/10)
    sv=sv.mean()
    Sv.append(10*np.log10(sv))

# plot a Sv(f) over school
fig1 = figure()
sv=plt.plot(f_m/1000,Sv) # values are for some reason to low, add ~17dB
plt.title('Sv(f) averaged over school depths')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Range [m]')
plt.grid()
plt.show()

# Store Sv(f) and f for further analysis
SvfOut = np.concatenate((f_m[np.newaxis],Sv_m_n), axis=0)
np.save('Svf.npy',SvfOut)