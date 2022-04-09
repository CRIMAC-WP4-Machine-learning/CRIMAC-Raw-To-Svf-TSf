
from Core.EK80DataContainer import EK80DataContainer
from plots import *

do_plot = True
file = r'./data/CRIMAC_SphereBeam.json'

data = EK80DataContainer(file)  # TS sphere


# Obtain data from raw file
# Estimate variable values at center frequency and other frequencies
# Generate ideal windowed transmit signal

# Unpack variables
z_td_e, f_s, n_f_points = data.cont.getParameters()

z_rx_e = data.trcv.getParameters()

f_0, f_1, f_c, tau, slope, sampleInterval, p_tx_e = data.parm.getParameters()

f_n, G_fnom, psi_f_n, angle_offset_alongship_fnom, \
angle_offset_athwartship_fnom, angle_sensitivity_alongship_fnom, \
angle_sensitivity_athwartship_fnom, beam_width_alongship_fnom, \
beam_width_alongship_fnom, corrSa = data.trdu.getParameters()

c, alpha, temperature, salinity, \
acidity, latitude, depth, dropKeelOffset = data.envr.getParameters()

filter_v, N_v = data.filt.getParameters()

offset, sampleCount, y_rx_nu, N_u, y_rx_nu = data.raw3.getParameters()

# Frequency vector for both TS and Sv (grid for index m)
f_m = np.linspace(f_0, f_1, n_f_points)

# Range vector
r_n, dr = data.calcRange(
    sampleInterval,
    sampleCount,
    c,
    offset)

# Absorption coefficient at center frequency and f_m
alpha_f_c = data.calc_alpha(f_c)
alpha_m = data.calc_alpha(f_m)

# Wavelength at center frequency and f_m
lambda_f_c = data.calc_lambda(f_c)
lambda_m = data.calc_lambda(f_m)

# Angle sensitivities at center frequency
gamma_theta_f_c = data.calc_gamma_alongship(f_c)
gamma_phi_f_c = data.calc_gamma_athwartship(f_c)

# On-axis gain for center frequency and f_m
g_0_f_c = data.calc_g(0, 0, f_c)
g_0_m = data.calc_g(0, 0, f_m)

# Two-way equivalent beam angle at center frequency and f_m
psi_f_c = Calculation.calcpsi(psi_f_n, f_n, f_c)
psi_m = Calculation.calcpsi(psi_f_n, f_n, f_m)

#
# Chapter I: Signal generation
#

# Ideal windowed transmit signal
y_tx_n, t = Calculation.generateIdealWindowedTransmitSignal(
    f_0, f_1, tau, f_s, slope)

# Plots for paper
if do_plot:
    plotytx(f_0, f_1, tau, f_s, y_tx_n, slope)

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
#
# The filter coefficients 'h_fl_iv' are accessible through 'data.filter_vn':
# filter_v[0]["h_fl_i"]
# filter_v[1]["h_fl_i"]
#
# the corresponding decimation factors are available through
# filter_v[0]["D"]
# filter_v[1]["D"]

# The decimated sampling frequency after each filter stage
f_s_dec_v = Calculation.calcDecmiatedSamplingRate(filter_v, f_s)

# The final decimated sampling frequency after last filter stage
f_s_dec = f_s_dec_v[-1]

# Plots for paper
if do_plot:
    plotfir(filter_v, f_s_dec_v, f_0, f_1)

#
# Chapter IID: Pulse compression
#

# Normalized ideal transmit signal
y_tilde_tx_n = Calculation.calcNormalizedTransmitSignal(y_tx_n)

# Passing the normalized ideal transmit signal through the filter bank
y_tilde_tx_nv = Calculation.calcFilteredAndDecimatedSignal(
    y_tilde_tx_n, filter_v)

# Use the normalized, filtered, and decimated transmit signal from the last
# filter stage as matched filter
y_mf_n = y_tilde_tx_nv[-1]

# Plots for paper
if do_plot:
    plotymfn(y_mf_n)

# Auto correlation function and effective pulse duration of the matched
# filter
y_mf_auto_n, tau_eff = Calculation.calcAutoCorrelation(y_mf_n, f_s_dec)

# Plots for paper
if do_plot:
    plotACF(y_mf_auto_n)

# Pulse compressed signals for each channel (transducer sector)
y_pc_nu = Calculation.calcPulseCompressedSignals(y_rx_nu, y_mf_n)

# Average signal over all channels (transducer sectors)
y_pc_n = Calculation.calcAverageSignal(y_pc_nu)

# Average signals over paired channels corresponding to transducer halves
# fore, aft, starboard, port
y_pc_halves_n = Calculation.calcTransducerHalves(y_pc_nu)

#
# Chapter IIE: Power angles and samples
#

# Total received power for all channels (all transducer sectors)
p_rx_e_n = Calculation.calcPower(
    y_pc_n,
    z_td_e,
    z_rx_e,
    N_u)

# Physical angles
theta_n, phi_n = Calculation.calcAngles(
    y_pc_halves_n,
    gamma_theta_f_c,
    gamma_phi_f_c)

# Plots for paper
if do_plot:
    plotThetaPhi(theta_n, phi_n)

#
# Chapter III: TARGET STRENGTH
#

# TS estimation parameters
r0 = 5.8 - 0.5
r1 = 5.8 + 0.5
before = 0.5
after = 1

# Point scattering strength (Sp)
Sp_n = Calculation.calcSp(
    p_rx_e_n,
    r_n,
    alpha_f_c,
    p_tx_e,
    lambda_f_c,
    g_0_f_c,
    r0,
    r1)

# Data from a single target (_t denotes data from single target)
r_t, theta_t, phi_t, y_pc_t_n, dum_p, dum_theta, dum_phi, dum_r = \
    Calculation.singleTarget(
        y_pc_n, p_rx_e_n, theta_n, phi_n, r_n,
        r0, r1, before, after)

# Reduced auto correlation signal
y_mf_auto_red_n = Calculation.alignAuto(y_mf_auto_n, y_pc_t_n)

# Plots for paper
if do_plot:
    plotSingleTarget(dum_r, dum_p, dum_theta, r_t, dum_phi, phi_t, y_mf_auto_red_n)

# DFT of target signal, DFT of reduced auto correlation signal, and
# normalized DFT of target signal
Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m = \
    Calculation.calcDFTforTS(y_pc_t_n, y_mf_auto_red_n,
                             n_f_points, f_m, f_s_dec)

# Received power spectrum for a single target
P_rx_e_t_m = Calculation.calcPowerFreqTS(
    N_u,
    Y_tilde_pc_t_m,
    z_td_e,
    z_rx_e)

# Transducer gain incorporating both on-axis gain and
# beam pattern based oon target bearing
g_theta_phi_m = data.calc_g(theta_t, phi_t, f_m)

# Target strength spectrum
TS_m = Calculation.calcTSf(
    P_rx_e_t_m, r_t, alpha_m, p_tx_e, lambda_m,
    g_theta_phi_m)

# Plots for paper

plotTS(f_m, Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m, g_theta_phi_m, TS_m)

if do_plot:
    plt.show()
