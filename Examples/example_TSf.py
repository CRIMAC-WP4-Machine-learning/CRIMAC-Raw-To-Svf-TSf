import matplotlib.pyplot as plt
import numpy as np
import argparse

from Core.EK80DataContainer import EK80DataContainer
from Core.Calculation import EK80CalculationPaper

# python example_TSf.py --file ..\Data\pyEcholabEK80data.json --r0 10 --r1 30 --before 0.5 --after 1
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do signal processing on ek80 data')
    parser.add_argument('--file', required=True, metavar='File', type=str,
                        help='File where data is stored')
    parser.add_argument('--r0', required=True, type=float, help='Start range in meters')
    parser.add_argument('--r1', required=True, type=float, help='End range in meters')
    parser.add_argument('--before', required=True, type=float, help='Range before target in meters')
    parser.add_argument('--after', required=True, type=float, help='Range after target in meters')

    args = parser.parse_args()

    # Test data
    data = EK80DataContainer('../data/pyEcholabEK80data.json')

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

     # Generate ideal send pulse
    y_tx_n, t = EK80CalculationPaper.generateIdealWindowedSendPulse(
        f0, f1, tau, f_s, slope)

    # The sampling freq for each filter step
    f_s_dec_v = EK80CalculationPaper.calcDecmiatedSamplingRate(filter_v, f_s)

    # The final sampling frequency
    f_s_dec = f_s_dec_v[-1]

    # The normalized ideal transmit signal
    y_tilde_tx_n = EK80CalculationPaper.calcNormalizedTransmitSignal(y_tx_n)

    # Passing the normalized and ideal transmit signal through the filter bank
    y_tilde_tx_nv = EK80CalculationPaper.calcFilteredAndDecimatedSignal(
        y_tilde_tx_n, filter_v)

    # Use the normalized, filtered and decimated transmit signal from the last
    # filter stage for the matched filter.
    y_mf_n = y_tilde_tx_nv[-1]

    # The autocorrelation function and efficient pulse duration of the mathced
    # filter
    y_mf_auto_n, tau_eff = EK80CalculationPaper.calcAutoCorrelation(
        y_mf_n, f_s_dec)

    # Calculating the pulse compressed quadrant signals separately on each channel
    y_pc_nu = EK80CalculationPaper.calcPulseCompressedQuadrants(y_rx_nu, y_mf_n)

    # Calculating the average signal over the channels
    y_pc_n = EK80CalculationPaper.calcAvgSumQuad(y_pc_nu)

    # Calculating the average signal over paired fore, aft, starboard, port channel
    y_pc_halves_n = EK80CalculationPaper.calcTransducerHalves(y_pc_nu)

    # Calcuate the power across transducer channels
    p_rx_e_n = EK80CalculationPaper.calcPower(y_pc_n, z_td_e, z_rx_e, N_u)

    # Calculate the angle sensitivities
    gamma_theta = EK80CalculationPaper.calcGamma(angle_sensitivity_alongship_fnom, f_c, fnom)
    gamma_phi = EK80CalculationPaper.calcGamma(angle_sensitivity_athwartship_fnom, f_c, fnom)

    # Calculate the physical angles
    theta_n, phi_n = EK80CalculationPaper.calcAngles(y_pc_halves_n, gamma_theta, gamma_phi)

    # Get parameters
    r0 = args.r0
    r1 = args.r1
    before = args.before
    after = args.after

    r_n, _ = EK80CalculationPaper.calcRange(sampleInterval, sampleCount, c, offset)

    alpha_f_c = EK80CalculationPaper.calcAbsorption(temperature, salinity, depth, acidity, c, f_c)

    # Calculate the point scattering strength (Sp)
    Sp_n = EK80CalculationPaper.calcSp(p_rx_e_n, r_n, alpha_f_c, p_tx_e, lambda_f_c, g_0_f_c, r0, r1)

    # Data from a single target (_t denotes data from single target)
    r_t, theta_t, phi_t, y_pc_t_n, dum_p, dum_theta, dum_phi, dum_r = \
        EK80CalculationPaper.singleTarget(
            y_pc_n, p_rx_e_n, theta_n, phi_n, r_n,
            r0, r1, before, after)

    # Pick the reduced samples from the mathed filtered decimated send pulse
    y_mf_auto_red_n = EK80CalculationPaper.alignAuto(y_mf_auto_n, y_pc_t_n)

    # DFT on the pulse compressed received signal and the pulse compressed
    # send pulse signal rediced mathed filtered
    Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m, f_m_t = \
        EK80CalculationPaper.calcDFTforTS(y_pc_t_n, y_mf_auto_red_n, n_f_points, f0, f1, f_s_dec)

    # Calculate the power by frequency from a single target
    P_rx_e_t_m = EK80CalculationPaper.calcPowerFreq(N_u, Y_tilde_pc_t_m, z_td_e, z_rx_e)

    # Calculate the target strength
    g_theta_phi_m = data.calc_g(theta_t, phi_t, f_m_t)
    lambda_m = data.calc_lambda_f(f_m_t)
    alpha_m = data.calc_alpha_f(f_m_t)
    TS_m = EK80CalculationPaper.calcTSf(P_rx_e_t_m, r_t, alpha_m, p_tx_e, lambda_m, g_theta_phi_m)

    plt.figure()
    plt.title('Tsf for target in layer from {:.1f} m to {:.1f} m'.format(args.r0, args.r1))
    plt.plot(TS_m)
    plt.xlabel('kHz')
    plt.ylabel('dB')

    plt.show()




