import matplotlib.pyplot as plt
import numpy as np
from Core.Calculation import Calculation


def plot_ytx(f_0, f_1, tau, f_s,y_tx_n,slope):

    # Example of ideal windowed transmit signal with slope 0.5
    y_tx_n05slope, t = Calculation.generateIdealWindowedTransmitSignal(
        f_0, f_1, tau, f_s, .5)

    plt.figure()
    plt.plot(t * 1000, y_tx_n, t * 1000, y_tx_n05slope)
    plt.title(
        'Ideal windowed transmit pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
            .format(f_0 / 1000, f_1 / 1000, slope))
    plt.xlabel('time (ms)')
    plt.ylabel('amplitude')
    plt.savefig('./Paper/Fig_ytx.png')


def plot_fir(filter_v,f_s_dec_v,f_0, f_1):
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
    plt.xlim([50000, 210000])
    plt.savefig('./Paper/Fig_fir.png')


def plot_y_mf_n(y_mf_n):
    plt.figure()
    plt.plot(np.abs(y_mf_n))
    plt.title('The absolute value of the filtered and decimated output signal')
    plt.xlabel('samples ()')
    plt.ylabel('amplitude')
    plt.savefig('./Paper/Fig_y_mf_n.png')


def plot_ACF(y_mf_auto_n):
    plt.figure()
    plt.plot(np.abs(y_mf_auto_n))
    plt.title('The autocorrelation function of the matched filter.')
    plt.xlabel('Samples')
    plt.ylabel('ACF')
    plt.savefig('./Paper/Fig_ACF.png')


def plot_theta_phi(theta_n,phi_n):
    # Plot angles
    plt.figure()
    plt.plot(theta_n)
    plt.plot(phi_n)
    plt.title('The physical angles.')
    plt.xlabel(' ')
    plt.ylabel('Angles')
    plt.savefig('./Paper/Fig_theta_phi.png')


def plt_single_target(dum_r, dum_p,dum_theta,r_t,dum_phi,phi_t,y_mf_auto_red_n):
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
    axs[2].set_xlabel('Range [m]')
    plt.savefig('./Paper/Fig_singleTarget.png')


def plot_TS(f_m,Y_pc_t_m,Y_mf_auto_red_m,Y_tilde_pc_t_m,g_theta_phi_m,TS_m):
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

    # Store TS(f) and f for further analysis
    # TSfOut = np.stack((f_m,TS_m), axis=0)
    # np.save('TSf.npy',TSfOut)


def plotSvf(f_m,Sv_m_n,svf_range):
    plt.figure()
    _f = f_m / 1000
    plt.imshow(Sv_m_n, extent=[_f[0], _f[-1], svf_range[-1], svf_range[0]], origin='upper',
               interpolation=None)
    plt.colorbar()
    plt.title('Echogram [Sv]')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Range [m]')
    plt.axis('auto')
    plt.savefig('./Paper/Fig_Sv_m_n.png')

    plt.figure()
    # Plot Sv(f) in one depth in the middle of layer
    indices=np.where(np.logical_and(svf_range>=15, svf_range<=34))
    Sv=[]
    plt.plot(Sv_m_n[int(len(indices[0]) / 2) - 1,])
    plt.title('Sv(f) at one depth')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Sv')
    plt.grid()


    indices = np.where(np.logical_and(svf_range >= 60, svf_range <= 70))
    # returns (array([3, 4, 5]),)
    Sv = []
    for i in range(len(f_m)):
        sv = 10 ** (Sv_m_n[indices, i] / 10)
        sv = sv.mean()
        Sv.append(10 * np.log10(sv))

    # plot a Sv(f) over school
    #from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap
    #fig1 = figure()
    plt.figure()
    plt.plot(f_m / 1000, Sv)  # values are for some reason to low, add ~17dB
    plt.title('Sv(f) averaged over school depths')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Range [m]')
    plt.grid()
    plt.savefig('./Paper/Fig_Sv_m.png')
    # Store Sv(f) and f for further analysis
    SvfOut = np.concatenate((f_m[np.newaxis],Sv_m_n), axis=0)
    # np.save('Svf.npy',SvfOut)
