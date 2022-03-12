import matplotlib.pyplot as plt
import numpy as np
import argparse

from Core.EK80DataContainer import EK80DataContainer
from Core.Calculation import Calculation

# python example_TSf.py --file ..\Data\pyEcholabEK80data.json --r0 10 --r1 30 --before 0.5 --after 1
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do signal processing on ek80 data')
    parser.add_argument('--r0', required=True, type=float, help='Start range in meters')
    parser.add_argument('--r1', required=True, type=float, help='End range in meters')
    parser.add_argument('--before', required=True, type=float, help='Range before target in meters')
    parser.add_argument('--after', required=True, type=float, help='Range after target in meters')

    args = parser.parse_args()

    # Test data
    data = EK80DataContainer('../Data/CRIMAC_SphereBeam.json')  # TS sphere

    # Unpack variabels
    z_td_e, f_s, n_f_points = data.cont.getParameters()

    z_rx_e = data.trcv.getParameters()

    f_0, f_1, f_c, tau, slope, sampleInterval, p_tx_e = data.parm.getParameters()

    # The frequency vector for both Ts and Sv (grid for index m)
    f_m = np.linspace(f_0, f_1, n_f_points)

    f_n, G_fnom, psi_f_n, angle_offset_alongship_fnom, \
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

    # TODO: Consider to use _f_m instead of _m, depending on the
    # way it will be written in the paper, for the following parameters:

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

    # Generate ideal send pulse
    y_tx_n, t = Calculation.generateIdealWindowedTransmitSignal(
        f_0, f_1, tau, f_s, slope)

    # The filter coefficients 'h_fl_iv' are accessible through 'data.filter_vn':
    filter_v[0]["h_fl_i"]
    filter_v[1]["h_fl_i"]

    # the corresponding decimation factors are available through
    filter_v[0]["D"]
    filter_v[1]["D"]

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
    y_pc_nu = Calculation.calcPulseCompressedSignals(y_rx_nu, y_mf_n)

    # Calculating the average signal over the channels
    y_pc_n = Calculation.calcAverageSignal(y_pc_nu)

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

    # Get parameters
    r0 = args.r0
    r1 = args.r1
    before = args.before
    after = args.after

    # Calculate the point scattering strength (Sp)
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

    # Pick the reduced samples from the mathed filtered decimated send pulse
    y_mf_auto_red_n = Calculation.alignAuto(y_mf_auto_n, y_pc_t_n)

    # DFT on the pulse compressed received signal and the pulse compressed
    # send pulse signal (reduced and matched filtered)
    Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m = \
        Calculation.calcDFTforTS(y_pc_t_n, y_mf_auto_red_n,
                                          n_f_points, f_m, f_s_dec)

    # Calculate the power by frequency from a single target
    P_rx_e_t_m = Calculation.calcPowerFreq(
        N_u,
        Y_tilde_pc_t_m,
        z_td_e,
        z_rx_e)

    # Calculate the target strength
    g_theta_phi_m = data.calc_g(theta_t, phi_t, f_m)

    TS_m = Calculation.calcTSf(
        P_rx_e_t_m, r_t, alpha_m, p_tx_e, lambda_m,
        g_theta_phi_m)

    np.savetxt('f_m.txt', f_m, fmt='%1.4e')
    np.savetxt('TS_m.txt', TS_m, fmt='%1.4e')

    plt.figure()
    plt.title('Tsf for target in layer from {:.1f} m to {:.1f} m'.format(args.r0, args.r1))
    plt.plot(f_m/1e3, TS_m)
    plt.xlabel('kHz')
    plt.ylabel('dB')

    plt.show()




