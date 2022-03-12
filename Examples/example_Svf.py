import matplotlib.pyplot as plt
import numpy as np
import argparse

from Core.EK80Calculation import EK80CalculationPaper

# python example_Svf.py --file ..\Data\pyEcholabEK80data.json --r0 10 --r1 30
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do signal processing on ek80 data')
    parser.add_argument('--r0', type=float, help='Start range in meters')
    parser.add_argument('--r1', type=float, help='End range in meters')

    args = parser.parse_args()

    # Test data
    data = EK80DataContainer('./data/CRIMAC_Svf.json')  # Sv school

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

    # Calculate Psi for f_c and on the f_m grid
    Psi_f_c = EK80CalculationPaper.calc_psi_f(psi_f_n, f_n, f_c)
    # TODO: double check this. I think it is ok:
    Psi_m = EK80CalculationPaper.calc_psi_f(psi_f_n, f_n, f_m)

    # Generate ideal send pulse
    y_tx_n, t = EK80CalculationPaper.generateIdealWindowedTransmitSignal(
        f_0, f_1, tau, f_s, slope)

    # The filter coefficients 'h_fl_iv' are accessible through 'data.filter_vn':
    filter_v[0]["h_fl_i"]
    filter_v[1]["h_fl_i"]

    # the corresponding decimation factors are available through
    filter_v[0]["D"]
    filter_v[1]["D"]

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
    y_pc_nu = EK80CalculationPaper.calcPulseCompressedSignals(y_rx_nu, y_mf_n)

    # Calculating the average signal over the channels
    y_pc_n = EK80CalculationPaper.calcAverageSignal(y_pc_nu)

    # Calcuate the power across transducer channels
    p_rx_e_n = EK80CalculationPaper.calcPower(
        y_pc_n,
        z_td_e,
        z_rx_e,
        N_u)

    # Get parameters
    r0 = args.r0
    r1 = args.r1

    #
    # Chapter IV: VOLUME BACKSCATTERING STRENGTH
    #

    # Calculate average Sv
    # TODO: I get zero power in the p_rx_e_n. Fails when doing log10. "Quickfix":
    p_rx_e_n = p_rx_e_n + .0000000000000001

    # TODO: Range equal to zero will not work. either remove first sample or
    # reconsider the range vector (log10(0) does not exist). Hack:
    r_n[r_n == 0] = 0.0000000001
    Sv_n = EK80CalculationPaper.calc_Sv(p_rx_e_n, r_n, lambda_f_c,
                                        p_tx_e, alpha_f_c, c, tau_eff,
                                        Psi_f_c, g_0_f_c)

    # Calculate the pulse compressed signal adjusted for spherical loss
    y_pc_s_n = EK80CalculationPaper.calc_PulseCompSphericalSpread(y_pc_n, r_n)

    # Hanning window
    w_tilde_i, N_w, t_w, t_w_n = EK80CalculationPaper.defHanningWindow(c, tau, dr,
                                                                       f_s_dec)

    # Calculate the DFT on the pulse compressed signal

    step = 1  # Needs some thoughts... 50% overlapp
    # Sjekk at n_f_ponts er 2 pulslengder, det er det vi vil ha
    Y_pc_v_m_n, Y_mf_auto_m, Y_tilde_pc_v_m_n, svf_range \
        = EK80CalculationPaper.calcDFTforSv(
        y_pc_s_n, w_tilde_i, y_mf_auto_n, N_w, n_f_points, f_m, f_s_dec,
        r_n, step)

    # Calculate the power
    P_rx_e_t_m_n = EK80CalculationPaper.calcPowerFreqforSv(
        Y_tilde_pc_v_m_n, N_u, z_rx_e, z_td_e)

    # Calculate the Sv(f)
    # TODO: Range == 0 does not work well with log10. another hack:

    Sv_m_n = EK80CalculationPaper.calcSvf(P_rx_e_t_m_n,
                                          alpha_m, p_tx_e, lambda_m, t_w,
                                          Psi_m, g_0_m, c, svf_range)



    #ekdata = EK80DataContainer(args.file)
    ekcalc = EK80CalculationPaper(args.file)

    y_pc_u = ekcalc.calcPulseCompressedSignals(ekcalc.y_rx_nu)
    y_pc = ekcalc.calcAverageSignal(y_pc_u)
    p_rx_e = ekcalc.calcPower(y_pc)

    Sv, r = ekcalc.calcSv(p_rx_e, args.r0, args.r1)
    Svf, rSvf, f = ekcalc.calcSvf(y_pc, args.r0, args.r1, overlap=0.5)

    investigate_r = [args.r0, args.r1]
    dr = rSvf[1]-rSvf[0]

    if len(investigate_r) > 1 and dr<args.r1-args.r0:
        ridx = np.where((rSvf >= investigate_r[0]) & (rSvf <= investigate_r[1]))[0]
    else:
        ridx = np.round(investigate_r / dr).astype(int)


    Svf_layer_lin = []
    nidx = 0
    for idx in ridx:
        nidx += 1
        if nidx == 1:
            Svf_layer_lin = 10**(Svf[idx, :] / 10)
        else:
            Svf_layer_lin = Svf_layer_lin + 10**(Svf[idx, :] / 10)

    Svf_layer = 10*np.log10(Svf_layer_lin / nidx)


    plt.figure()
    ax1 = plt.gca()
    ax1.imshow(Svf.T, extent=[rSvf[0], rSvf[-1], f[0] / 1000, f[-1] / 1000],origin='lower')
    ax1.axis('auto')
    ax1.set_ylabel('kHz')
    plt.xlabel('Range(m)')
    ax2 = ax1.twinx()
    ax2.plot(r, Sv,'k', alpha=0.3)
    ax2.set_ylabel('Sv(dB)')
    ax2.axis('auto')
    plt.title('Spectrogram Sv overlayed')

    plt.figure()
    plt.title('Svf@{:.1f}m to {:.1f}m'.format(rSvf[ridx[0]], rSvf[ridx[-1]]))
    for idx in ridx:
        plt.plot(f/1000, Svf[idx, :], label='{:0.1f}m'.format(rSvf[idx]))
    plt.xlabel('kHz')
    plt.ylabel('dB')
    plt.legend()

    plt.figure()
    plt.title('Svf for layer from {:.1f} m to {:.1f} m'.format(args.r0, args.r1))
    plt.plot(f/1000, Svf_layer)
    plt.xlabel('kHz')
    plt.ylabel('dB')

    plt.show()



