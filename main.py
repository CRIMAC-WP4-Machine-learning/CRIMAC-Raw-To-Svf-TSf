import os.path

import argparse
from Core.EK80DataContainer import EK80DataContainer
from plots import *


def preCalculations(data):
    """
    Reads echosounder data, single ping, from datafile (json), performs
    pre-calculations.
    
    Parameters
    ----------
    
    Returns
    -------
    All input data for further calculations.
    
    
    """

    global z_td_e, z_rx_e, N_u
    global f_0, f_1, tau, f_s, slope, p_tx_e
    global filter_v, f_s_dec_v
    global y_rx_nu
    global lambda_f_c, alpha_f_c, gamma_theta_f_c, gamma_phi_f_c, g_0_f_c, psi_f_c
    global n_f_points, f_m, lambda_m, alpha_m, g_0_m, psi_m
    global r_n, dr, c

    # Obtain data from raw file
    # Estimate variable values at center frequency and other frequencies
    # Generate ideal windowed transmit signal

    # Unpack variables
    z_td_e, f_s, n_f_points = data.cont.getParameters()

    z_rx_e = data.trcv.getParameters()

    f_0, f_1, f_c, tau, slope, sampleInterval, p_tx_e = data.parm.getParameters()

    (
        f_n,
        G_fnom,
        psi_f_n,
        angle_offset_alongship_fnom,
        angle_offset_athwartship_fnom,
        angle_sensitivity_alongship_fnom,
        angle_sensitivity_athwartship_fnom,
        beam_width_alongship_fnom,
        beam_width_alongship_fnom,
        corrSa,
    ) = data.trdu.getParameters()

    (
        c,
        alpha,
        temperature,
        salinity,
        acidity,
        latitude,
        depth,
        dropKeelOffset,
    ) = data.envr.getParameters()

    filter_v, N_v = data.filt.getParameters()

    offset, sampleCount, y_rx_nu, N_u, y_rx_nu = data.raw3.getParameters()

    # Frequency vector for both TS and Sv (grid for index m)
    f_m = np.linspace(f_0, f_1, n_f_points)

    # Range vector
    r_n, dr = data.calcRange(sampleInterval, sampleCount, c, offset)

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


def calcBasics(do_plot):
    """
    XXX.
    
    Parameters
    ----------
    
    """

    global f_s_dec_v, f_s_dec
    global y_tx_n, y_mf_n, y_mf_auto_n, tau_eff
    global y_pc_n, p_rx_e_n, theta_n, phi_n

    #
    # Chapter I: Signal generation
    #

    # Ideal windowed transmit signal
    y_tx_n, t = Calculation.generateIdealWindowedTransmitSignal(
        f_0, f_1, tau, f_s, slope
    )

    # Plots for paper
    # if do_plot:
    #     plotytx(f_0, f_1, tau, f_s, y_tx_n, slope)

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
    y_tilde_tx_nv = Calculation.calcFilteredAndDecimatedSignal(y_tilde_tx_n, filter_v)

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
    p_rx_e_n = Calculation.calcPower(y_pc_n, z_td_e, z_rx_e, y_pc_nu.shape[0])

    # Physical angles
    theta_n, phi_n = Calculation.calcAngles(
        y_pc_halves_n, gamma_theta_f_c, gamma_phi_f_c
    )

    # Plots for paper
    if do_plot:
        plotThetaPhi(theta_n, phi_n, dr)


def calcTS():
    """
    XXX.
    
    Parameters
    ----------
    
    """

    global f_m, Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m, g_theta_phi_m, TS_m
    global r_t, theta_t, phi_t, y_pc_t_n, dum_p, dum_theta, dum_phi, dum_r
    global y_mf_auto_red_n

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
        p_rx_e_n, r_n, alpha_f_c, p_tx_e, lambda_f_c, g_0_f_c, r0, r1
    )

    # Data from a single target (_t denotes data from single target)
    (
        r_t,
        theta_t,
        phi_t,
        y_pc_t_n,
        dum_p,
        dum_theta,
        dum_phi,
        dum_r,
    ) = Calculation.singleTarget(
        y_pc_n, p_rx_e_n, theta_n, phi_n, r_n, r0, r1, before, after
    )

    # Reduced auto correlation signal
    y_mf_auto_red_n = Calculation.alignAuto(y_mf_auto_n, y_pc_t_n)

    # Plots for paper
    plotSingleTarget(dum_r, dum_p, dum_theta, r_t, dum_phi, phi_t, y_mf_auto_red_n, y_pc_t_n)

    # DFT of target signal, DFT of reduced auto correlation signal, and
    # normalized DFT of target signal
    Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m = Calculation.calcDFTforTS(
        y_pc_t_n, y_mf_auto_red_n, n_f_points, f_m, f_s_dec
    )

    # Received power spectrum for a single target
    P_rx_e_t_m = Calculation.calcPowerFreqTS(N_u, Y_tilde_pc_t_m, z_td_e, z_rx_e)

    # Transducer gain incorporating both on-axis gain and
    # beam pattern based oon target bearing
    g_theta_phi_m = data.calc_g(theta_t, phi_t, f_m)

    # Target strength spectrum
    TS_m = Calculation.calcTSf(
        P_rx_e_t_m, r_t, alpha_m, p_tx_e, lambda_m, g_theta_phi_m
    )

    # Plots for paper
    plotTS(f_m, Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m, g_theta_phi_m, TS_m)


def calcSv():
    """
    XXX.
    
    Parameters
    ----------
    
    """

    global p_rx_e_n, f_m, svf_range, Sv_m_n

    #
    # Chapter IV: VOLUME BACKSCATTERING STRENGTH
    #

    # Sv estimation parameters
    step = 1  # Step in samples for sliding window

    # Volume backscattering strength compressed frequency band

    Sv_n = Calculation.calcSv(
        p_rx_e_n, r_n, lambda_f_c, p_tx_e, alpha_f_c, c, tau_eff, psi_f_c, g_0_f_c
    )

    # Pulse compressed signal adjusted for spherical loss
    y_pc_s_n = Calculation.calcPulseCompSphericalSpread(y_pc_n, r_n)

    # Hanning window
    w_tilde_i, N_w, t_w, t_w_n = Calculation.defHanningWindow(c, tau, dr, f_s_dec)

    # TODO: Currently step=1. Consider changing overlap.
    # Normalized DFT of sliding window data
    Y_pc_v_m_n, Y_mf_auto_m, Y_tilde_pc_v_m_n, svf_range = Calculation.calcDFTforSv(
        y_pc_s_n, w_tilde_i, y_mf_auto_n, N_w, n_f_points, f_m, f_s_dec, r_n, step
    )

    # Received power spectrum for sliding window
    P_rx_e_t_m_n = Calculation.calcPowerFreqSv(Y_tilde_pc_v_m_n, N_u, z_rx_e, z_td_e)

    # Volume backscattering strength spectrum for each range step
    Sv_m_n = Calculation.calcSvf(
        P_rx_e_t_m_n, alpha_m, p_tx_e, lambda_m, t_w, psi_m, g_0_m, c, svf_range
    )

    # Plots for paper
    plotSvf(f_m, Sv_m_n, svf_range)


# Example Usage :
#   main.py --tfile ./data/CRIMAC_SphereBeam.json --sfile ./data/CRIMAC_Svf.json
#   main.py --tfile ./data/CRIMAC_SphereBeam.json --sfile ./data/CRIMAC_Svf.json --plots false
#   main.py --tfile ./data/CRIMAC_SphereBeam.json --sfile none
if __name__ == "__main__":
    # Handle command line parameters
    #
    ap = argparse.ArgumentParser(description="Sv and TS calculations")
    ap.add_argument(
        "--sfile",
        default="./data/CRIMAC_Svf.json",
        help="File containing raw data for Sv calculation",
    )
    ap.add_argument(
        "--tfile",
        default="./data/CRIMAC_SphereBeam.json",
        help="File containing raw data for TS calculation",
    )
    ap.add_argument(
        "--plots", choices=["true", "false"], default="true", help="Show plots or not"
    )

    args = ap.parse_args()

    show_plots = False
    has_TS_file = False
    has_Sv_file = False

    if args.plots == "true":
        show_plots = True

    if os.path.exists(args.tfile):
        has_TS_file = True

    if os.path.exists(args.sfile):
        has_Sv_file = True

    # Do calculation according to commandline parameters
    #
    if has_TS_file:
        data = EK80DataContainer(args.tfile)  # TS sphere
        preCalculations(data)
        calcBasics(do_plot=show_plots)
        calcTS()

    if has_Sv_file:
        data = EK80DataContainer(args.sfile)
        preCalculations(data)
        calcBasics(do_plot=not has_TS_file and show_plots)
        calcSv()

    if (has_TS_file or has_Sv_file) and show_plots:
        plt.show()
