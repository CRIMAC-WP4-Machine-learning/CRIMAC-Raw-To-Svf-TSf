import matplotlib.pyplot as plt
import numpy as np
from Core.Calculation import Calculation
from Core.EK80DataContainer import EK80DataContainer


def preCalculations(data):

    global f_0,f_1, tau, f_s, slope, filter_v, z_td_e, z_rx_e, N_u
    global angle_sensitivity_alongship_fnom,angle_sensitivity_athwartship_fnom
    global f_c, f_n, y_rx_nu,  r_n, alpha_f_c, p_tx_e, lambda_f_c, g_0_f_c
    global n_f_points, f_m, alpha_m, p_tx_e, lambda_m, c, tau_eff, Psi_f_c
    global Psi_m, g_0_m, dr

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
    Psi_f_c = Calculation.calc_psi_f(Psi_f_n, f_n, f_c)
    Psi_m = Calculation.calc_psi_f(Psi_f_n, f_n, f_m)

def calcTS():
    y_tx_n, t = Calculation.generateIdealWindowedTransmitSignal(
        f_0, f_1, tau, f_s, slope)

    y_tx_n05slope, t = Calculation.generateIdealWindowedTransmitSignal(
        f_0, f_1, tau, f_s, .5)

    plt.figure()
    plt.plot(t * 1000, y_tx_n, t * 1000, y_tx_n05slope)
    plt.title(
        'Ideal enveloped send pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
            .format(f_0 / 1000, f_1 / 1000, slope))
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

    # The frequency response function of the filter is given by its
    # discrete time fourier transform:
    H0 = np.fft.fft(filter_v[0]["h_fl_i"])
    H1 = np.fft.fft(filter_v[1]["h_fl_i"])

    # Plot of the frequency response of the filters (power) (in dB)
    F0 = np.arange(len(H0)) * f_s_dec_v[0] / (len(H0))
    F1 = np.arange(len(H1)) * f_s_dec_v[1] / (len(H1))
    G0 = 20 * np.log10(np.abs(H0))
    # Repeat pattern for the second filter (4 times)
    F1l = np.append(F1, F1 + f_s_dec_v[1])
    F1l = np.append(F1l, F1 + 2 * f_s_dec_v[1])
    F1l = np.append(F1l, F1 + 3 * f_s_dec_v[1])
    G1 = 20 * np.log10(np.abs(H1))
    G1l = np.append(G1, G1)
    G1l = np.append(G1l, G1)
    G1l = np.append(G1l, G1)

    plt.figure()
    plt.plot(F0, G0,
             F1l, G1l,
             [f_0, f_1], [-140, -140])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.xlim([0, 100000])
    plt.savefig('./Paper/Fig_fir.png')

    #
    # Chapter IID: Pulse compression
    #

    # The normalized ideal transmit signal
    y_tilde_tx_n = Calculation.calcNormalizedTransmitSignal(y_tx_n)

    # Passing the normalized and ideal transmit signal through the filter bank
    y_tilde_tx_nv = Calculation.calcFilteredAndDecimatedSignal(
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
    y_mf_auto_n, tau_eff = Calculation.calcAutoCorrelation(
        y_mf_n, f_s_dec)

    plt.figure()
    plt.plot(np.abs(y_mf_auto_n))
    plt.title('The autocorrelation function of the matched filter.')
    plt.xlabel('samples')
    plt.ylabel('ACF')
    plt.savefig('./Paper/Fig_ACF.png')

    # Calculating the pulse compressed quadrant signals separately on each channel
    y_pc_nu = Calculation.calcPulseCompressedSignals(y_rx_nu, y_mf_n)

    # Calculating the average signal over the channels
    y_pc_n = Calculation.calcAverageSignal(y_pc_nu)

    # Calculating the average signal over paired fore, aft, starboard, port channel
    y_pc_halves_n = Calculation.calcTransducerHalves(y_pc_nu)

    #
    # Chapter IIE: Power angles and samples
    #

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
    # 'TS_Sphere_Beam':
    r0 = 5.8 - 0.5
    r1 = 5.8 + 0.5
    before = 0.5
    after = 1

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

    fig, axs = plt.subplots(3)
    fig.suptitle('Single target')
    axs[0].plot(dum_r, dum_p)
    axs[0].set_ylabel('Power')
    line1, = axs[1].plot(dum_r, dum_theta, label='$\\theta$')
    axs[1].plot([r_t, r_t], [-2, 2])
    line2, = axs[1].plot(dum_r, dum_phi, label='$\phi$')
    axs[1].plot(r_t, phi_t)
    axs[1].legend(handles=[line1, line2])
    axs[1].set_ylabel('Angles')
    axs[2].plot(dum_r, np.abs(y_mf_auto_red_n))
    axs[2].set_ylabel(' ')
    axs[2].set_xlabel('range (m)')
    plt.savefig('./Paper/Fig_singleTarget.png')

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
    """
    g_theta_t_phi_t_f_t = data.calc_B_theta_phi_m(theta_t, phi_t, f_m)
    G0_m = data.calc_G0_m(f_m)
    G_theta_phi_m = G0_m - g_theta_t_phi_t_f_t
    """
    g_theta_phi_m = data.calc_g(theta_t, phi_t, f_m)

    TS_m = Calculation.calcTSf(
        P_rx_e_t_m, r_t, alpha_m, p_tx_e, lambda_m,
        g_theta_phi_m)

    fig, axs = plt.subplots(5)
    axs[0].plot(f_m, np.abs(Y_pc_t_m))
    axs[0].set_ylabel('Y_tilde_pc_t_m')
    axs[1].plot(f_m, np.abs(Y_mf_auto_red_m))
    axs[1].set_ylabel('Y_mf_auto_red_m')
    axs[2].plot(f_m, np.abs(Y_tilde_pc_t_m))
    axs[2].set_ylabel('Y_tilde_pc_t_m')
    axs[3].plot(f_m,
                g_theta_phi_m)  # weird gain might be tracked down to  xml['angle_offset_alongship'] and xml['angle_offset_alongship']
    axs[3].set_ylabel('gain')
    axs[4].plot(f_m, TS_m)
    axs[4].set_xlabel('f (Hz)')
    axs[4].set_ylabel('TS(f)')
    plt.savefig('./Paper/Fig_TS.png')

def calcSv():
    # Generate ideal send pulse
    y_tx_n, t = Calculation.generateIdealWindowedTransmitSignal(
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


    _f=f_m / 1000
    plt.imshow(Sv_m_n, extent=[_f[0], _f[-1],svf_range[0], svf_range[-1]],origin='lower', vmin=-180, vmax=-120,interpolation=None)
    plt.colorbar()
    plt.title('Echogram [Sv]')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Range [m]')
    plt.show()

    plt.plot(Sv_m_n[7000,])

    indices = np.where(np.logical_and(svf_range >= 60, svf_range <= 70))
    # returns (array([3, 4, 5]),)
    Sv = []
    for i in range(len(f_m)):
        sv = 10 ** (Sv_m_n[indices, i] / 10)
        sv = sv.mean()
        Sv.append(10 * np.log10(sv))

    # plot a Sv(f) over school
    from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap
    fig1 = figure()
    sv = plt.plot(f_m / 1000, Sv)  # values are for some reason to low, add ~17dB
    plt.title('Sv(f) averaged over school depths')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Range [m]')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    data = EK80DataContainer('./data/CRIMAC_Svf.json')
    preCalculations(data)
    calcSv()

    data = EK80DataContainer('./data/CRIMAC_SphereBeam.json')  # TS sphere
    preCalculations(data)
    calcTS()