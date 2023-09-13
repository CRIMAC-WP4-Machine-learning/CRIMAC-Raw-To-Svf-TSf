import numpy as np

from Core.EK80DataContainer import EK80DataContainer


class Calculation(EK80DataContainer):
    #
    # Chapter IIB: Signal generation
    #

    @staticmethod
    def generateIdealWindowedTransmitSignal(f0, f1, tau, fs, slope):
        """
        Generate the ideal windowed transmit signal.
        
        Parameters
        ----------
        f0 : float
            The start frequency [Hz]
        f1 : float
            The end frequency [Hz]
        tau : float
            The transmit signal duration [s]
        fs : float
            The signal sample rate [Hz]
        slope : float
            The proportion of the pulse that is windowed [1]. This includes the start and end parts of the windowing.
                 
        Returns
        -------
        y : np.array
            The ideal windowed transmit signal amplitude [1]
        t : np.arrayfloat
            The time of each value in y [s]
    
        """
        
        nsamples = int(np.floor(tau * fs))
        t = np.linspace(0, nsamples - 1, num=nsamples) * 1 / fs
        y = Calculation.chirp(t, f0, tau, f1)
        L = int(np.round(tau * fs * slope * 2.0))  # Length of hanning window
        w = Calculation.hann(L)
        N = len(y)
        w1 = w[0 : int(len(w) / 2)]
        w2 = w[int(len(w) / 2) :]
        i0 = 0
        i1 = len(w1)
        i2 = N - len(w2)
        i3 = N
        y[i0:i1] = y[i0:i1] * w1
        y[i2:i3] = y[i2:i3] * w2

        return y, t

    @staticmethod
    def chirp(t, f0, t1, f1):
        """
        Generate a chirp pulse.
        
        Parameters
        ----------
        t : float
            Times at which to calculate the chirp signal [s]
        f0 : float
            Start frequency of the pulse [Hz]
        t1 : float
            Duration of the pulse [s]
        f1 : float
            End frequency of the pulse [Hz]
            
        Returns
        -------
        np.array
            The chirp pulse amplitude at times t [1]
        """
        
        a = np.pi * (f1 - f0) / t1
        b = 2 * np.pi * f0

        return np.cos(a * t * t + b * t)

    @staticmethod
    def hann(L):
        """
        Generate Hann window weights.
        
        Parameters
        ----------
        L : int
            The number of samples to use [1]
        
        Returns
        -------
        np.array
            The Hann window weights [1]
        """
        
        n = np.arange(0, L, 1)

        return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (L - 1)))

    #
    # Chapter IIC: Signal reception
    #

    @staticmethod
    def calcDecmiatedSamplingRate(filter_v, f_s):
        """
        Calculate the decimated sample rate.
        
        Parameters
        ----------
        filter_v : XXX
            XXX
        f_s : float
            The undecimated sample rate [Hz]
            
        Returns
        -------
        float
            XXX
        """
        
        f_s_dec = [f_s]
        if filter_v is not None:
            for v, _filter_v in enumerate(filter_v):
                f_s_dec.append(f_s_dec[v] / _filter_v["D"])

        return f_s_dec

    #
    # Chapter IID: Pulse compression
    #

    @staticmethod
    def calcNormalizedTransmitSignal(y_tx_n):
        """
        Normalise the transmit signal by the maximum of the transmit signal.
        
        Parameters
        ----------
        y_tx_n : np.array
            The transmit signal [V]
        
        Returns
        -------
        np.array
            The normalised transmit signal [1]
            
        """
        
        return y_tx_n / np.max(y_tx_n)

    @staticmethod
    def calcFilteredAndDecimatedSignal(y_tilde_tx_n, filter_v):
        """
        Filter and decimate a signal.
        
        Parameters
        ----------
        y_tilde_tx_n : np.array
            Normalised transmit signal [1]
        filter_v : XXX
            XXX
            
        Returns
        -------
        np.array
            The filtered and decimated transmit signal [1]
    """
        
        # Initialize with normalized transmit pulse
        y_tilde_tx_nv = [y_tilde_tx_n]
        v = 0

        if filter_v is not None:
            for filter_vi in filter_v:
                tmp = np.convolve(y_tilde_tx_nv[v], filter_vi["h_fl_i"], mode="full")[
                    0 :: filter_vi["D"]
                ]
                y_tilde_tx_nv.append(tmp)
                v += 1

        return y_tilde_tx_nv

    @staticmethod
    def calcAutoCorrelation(y_mf_n, f_s_dec):
        """
        Calculate the autocorrelation of the matched filter signal.
        
        Parameters
        ----------
        y_mf_n : np.array
            Input signal [1]
        f_s_dec : float
            Decimated sampling frequency [Hz]
            
        Returns
        -------
        y_mf_auto_n : np.array
            Autocorrelation of the input filter [1]
        t_eff : float
            Effective transmit pulse duration [s]
        """
        
        y_mf_n_conj_rev = np.conj(y_mf_n)[::-1]
        y_mf_twoNormSquared = np.linalg.norm(y_mf_n, 2) ** 2
        y_mf_n_conj_rev = y_mf_n_conj_rev
        y_mf_twoNormSquared = y_mf_twoNormSquared

        # Calculate auto correlation function for matched filter
        y_mf_auto_n = np.convolve(y_mf_n, y_mf_n_conj_rev) / y_mf_twoNormSquared

        # Estimate effective pulse duration
        p_tx_auto = np.abs(y_mf_auto_n) ** 2
        tau_eff = np.sum(p_tx_auto) / ((np.max(p_tx_auto)) * f_s_dec)

        return y_mf_auto_n, tau_eff

    @staticmethod
    def calcPulseCompressedSignals(quadrant_signals, y_mf_n):
        """
        Pulse compress input signals.
        
        Parameters
        ----------
        quadrant_signals: real
            The XXX
        y_mf_n: np.array:
            The XXX
            
        Returns
        -------
        np.array
           The XXX
        """
        
        # Do pulse compression on all quadrants
        y_mf_n_conj_rev = np.conj(y_mf_n)[::-1]
        y_mf_twoNormSquared = np.linalg.norm(y_mf_n, 2) ** 2

        pulseCompressedQuadrants = []
        start_idx = len(y_mf_n_conj_rev) - 1
        for u in quadrant_signals:
            # Pulse compression per quadrant
            y_pc_nu = np.convolve(y_mf_n_conj_rev, u, mode="full") / y_mf_twoNormSquared
            # Correct sample indexes for mached filter
            y_pc_nu = y_pc_nu[start_idx::]
            pulseCompressedQuadrants.append(y_pc_nu)

        return np.array(pulseCompressedQuadrants)

    @staticmethod
    def calcAverageSignal(y_pc_nu):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        
        return np.sum(y_pc_nu, axis=0) / y_pc_nu.shape[0]

    @staticmethod
    def calcTransducerHalves(y_pc_nu):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        y_pc_fore_n = 0.5 * (y_pc_nu[2, :] + y_pc_nu[3, :])
        y_pc_aft_n = 0.5 * (y_pc_nu[0, :] + y_pc_nu[1, :])
        y_pc_star_n = 0.5 * (y_pc_nu[0, :] + y_pc_nu[3, :])
        y_pc_port_n = 0.5 * (y_pc_nu[1, :] + y_pc_nu[2, :])

        return y_pc_fore_n, y_pc_aft_n, y_pc_star_n, y_pc_port_n

    #
    # Chapter IIE: Power angles and samples
    #

    @staticmethod
    def calcPower(y_pc, z_td_e, z_rx_e, nu):
        """
        Calculate the received power into a matched load.
        
        Output received power values of 0.0 are set to 1e-20.
        
        Parameters
        ----------
        y_pc : np.array
            Pulse compressed signal [V]
        z_td_e : float
            Transducer electrical impedance [Ω]
        z_rx_e :
            Receiver electrical impedance [Ω]
        nu : int
            Number of receiver channels [1]
        
        Returns
        -------
        np.array
            Received electrical power [W]
            
        """
        K1 = nu / ((2 * np.sqrt(2)) ** 2)
        K2 = (np.abs(z_rx_e + z_td_e) / z_rx_e) ** 2
        K3 = 1.0 / np.abs(z_td_e)
        C1Prx = K1 * K2 * K3
        Prx = C1Prx * np.abs(y_pc) ** 2
        Prx[Prx == 0] = 1e-20

        return Prx

    @staticmethod
    def calcAngles(y_pc_halves, gamma_theta, gamma_phi):
        """
        Calculate splitbeam angles from 4 quadrant receiver/transducer data.
        
        Parameters
        ----------
        y_pc_halves : np.array
            Pulse compressed signal from transducer halves [V]
        gamma_theta : float
             Major axis conversion factor from electrical splitbeam angles to physical angles []
        gamma_phi : float
            Minor axis conversion factor from electrical splitbeam angles to physical angles []
        
        Returns
        -------
        theta_n : np.array
            Major axis physical splitbeam angles [°]
        phi_n : np.array
            Minor axis physical splitbeam angles [°]
        
        """

        y_pc_fore_n, y_pc_aft_n, y_pc_star_n, y_pc_port_n = y_pc_halves

        y_theta_n = y_pc_fore_n * np.conj(y_pc_aft_n)
        y_phi_n = y_pc_star_n * np.conj(y_pc_port_n)

        theta_n = (
            np.arcsin(np.arctan2(np.imag(y_theta_n), np.real(y_theta_n)) / gamma_theta)
            * 180
            / np.pi
        )
        phi_n = (
            np.arcsin(np.arctan2(np.imag(y_phi_n), np.real(y_phi_n)) / gamma_phi)
            * 180
            / np.pi
        )

        return theta_n, phi_n

    #
    # Chapter III: TARGET STRENGTH
    #

    @staticmethod
    def calcSp(p_rx_e_n, r_n, alpha_f_c, p_tx_e, lambda_f_c, g_0_f_c, r0=None, r1=None):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # Pick a range of the data
        if r0 is not None and r1 is not None:
            Idx = np.where((r_n >= r0) & (r_n <= r1))
            r_n = r_n[Idx]
            p_rx_e_n = p_rx_e_n[Idx]

        # Calculate Sp for the choosen range
        S_p_n = (
            10.0 * np.log10(p_rx_e_n)
            + 40.0 * np.log10(r_n)
            + 2.0 * alpha_f_c * r_n
            - 10
            * np.log10((p_tx_e * lambda_f_c**2 * g_0_f_c**2) / (16.0 * np.pi**2))
        )

        return S_p_n

    @staticmethod
    def singleTarget(
        y_pc_n, p_rx_e_n, theta_n, phi_n, r_n, r0, r1, before=0.5, after=0.5
    ):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # This is a pseudo single target detector (SED) using the max peak
        # power within the range interval [r0 r1] as the single target
        # detection criteria

        # Get the samples within the sub range defined by the range interval
        if r0 is not None and r1 is not None:
            Idx = np.where((r_n >= r0) & (r_n <= r1))
            r_n_sub = r_n[Idx]
            y_pc_n_sub = y_pc_n[Idx]
            p_rx_e_n_sub = p_rx_e_n[Idx]
            theta_n_sub = theta_n[Idx]
            phi_n_sub = phi_n[Idx]

        # Use peak power within given range limits as index for single target
        # to get the range, theta and phi for the single target
        idx_peak_p_rx = np.argmax(p_rx_e_n_sub)
        r_t = r_n_sub[idx_peak_p_rx]
        theta_t = theta_n_sub[idx_peak_p_rx]
        phi_t = phi_n_sub[idx_peak_p_rx]

        # Extract pulse compressed samples "before" and "after" the peak power
        r_t_begin = r_t - before
        r_t_end = r_t + after
        Idx2 = np.where((r_n_sub >= r_t_begin) & (r_n_sub <= r_t_end))
        y_pc_t = y_pc_n_sub[Idx2]
        p_rx_t = p_rx_e_n_sub[Idx2]
        dum_theta = theta_n_sub[Idx2]
        dum_phi = phi_n_sub[Idx2]
        dum_r = r_n_sub[Idx2]

        return r_t, theta_t, phi_t, y_pc_t, p_rx_t, dum_theta, dum_phi, dum_r

    @staticmethod
    def alignAuto(y_mf_auto_n, y_pc_t_n):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # The equivalent samples around the peak in the decimated tx signal
        idx_peak_auto = np.argmax(np.abs(y_mf_auto_n))
        idx_peak_y_pc_t_n = np.argmax(np.abs(y_pc_t_n))

        left_samples = idx_peak_y_pc_t_n
        right_samples = len(y_pc_t_n) - idx_peak_y_pc_t_n

        idx_start_auto = max(0, idx_peak_auto - left_samples)
        idx_stop_auto = min(len(y_mf_auto_n), idx_peak_auto + right_samples)

        y_mf_auto_red_n = y_mf_auto_n[idx_start_auto:idx_stop_auto]

        return y_mf_auto_red_n

    @staticmethod
    def calcDFTforTS(y_pc_t_n, y_mf_auto_red_n, n_f_points, f_m, f_s_dec):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # The number of DFT points inpower of 2
        N_DFT = int(2 ** np.ceil(np.log2(n_f_points)))

        idxtmp = np.floor(f_m / f_s_dec * N_DFT).astype("int")
        idx = np.mod(idxtmp, N_DFT)

        # DFT for the target signal
        _Y_pc_t_m = np.fft.fft(y_pc_t_n, n=N_DFT)
        Y_pc_t_m = _Y_pc_t_m[idx]

        # DFT for the transmit signal
        _Y_mf_auto_red_m = np.fft.fft(y_mf_auto_red_n, n=N_DFT)
        Y_mf_auto_red_m = _Y_mf_auto_red_m[idx]

        # The Normalized DFT
        Y_tilde_pc_t_m = Y_pc_t_m / Y_mf_auto_red_m

        return Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m

    @staticmethod
    def calcPowerFreqTS(N_u, Y_tilde_pc_t_m, z_td_e, z_rx_e):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        imp = (np.abs(z_rx_e + z_td_e) / np.abs(z_rx_e)) ** 2 / np.abs(z_td_e)
        P_rx_e_t_m = N_u * (np.abs(Y_tilde_pc_t_m) / (2 * np.sqrt(2))) ** 2 * imp
        return P_rx_e_t_m

    def calcg0(self, f):
        """
        Calculate gain at given frequencies.
        
        Uses calibration data if available, otherwise uses a theoretical 
        variation of gain with frequency relationship and nominal gain at
        a nominal frequency.
        
        Parameters
        ----------
        f : np.array
            Frequencies to calculate the gain at [Hz]
        
        Returns
        -------
        np.array
            Gain values at the given frequencues, `f` [dB]
        
        """
        if self.isCalibrated:
            # Calibrated case
            return np.interp(f, self.frqp.frequencies, self.frqp.gain)
        else:
            # Uncalibrated case
            return self.G_fnom + 20 * np.log10(f / self.fnom)

    @staticmethod
    def calcg0_calibrated(f, freq, gain):
        """
        Interpate frequency dependent gain data onto specified frequencies.
        
        Uses one-dimensional piecewise linear interpolation.
        
        Parameters
        ----------
        f : np.array
            Frequencies to interpolate on to [Hz]
        freq : np.array
            Frequencies at which gain values are provided [Hz]
        gain : np.array
            Gain values to use in the interpolation [dB]
        
        Returns
        -------
        np.array
            The gain values at frequencies given in `f` [1]
        
        """
        dB_G0 = np.interp(f, freq, gain)

        return np.power(10, dB_G0 / 10)

    @staticmethod
    def calcg0_notcalibrated(f, f_n, G_f_n):
        """
        Estimate frequency dependent gain based on theoretical variation with frequency.
        
        Parameters
        ----------
        f : np.array
            Frequencies at which to calculate the gain [Hz]
        f_n : float
            Frequency that the given gain is for [Hz]
        G_f_n : float
            Gain at `f_n` [dB]
        
        Returns
        -------
        np.array
            The gain values at frequencies given in `f` [1]
        """
        dB_G0 = G_f_n + 20 * np.log10(f / f_n)

        return np.power(10, dB_G0 / 10)

    @staticmethod
    def calcTSf(P_rx_e_t_m, r_t, alpha_m, p_tx_e, lambda_m, g_theta_t_phi_t_f_t):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        TS_m = (
            10 * np.log10(P_rx_e_t_m)
            + 40 * np.log10(r_t)
            + 2 * alpha_m * r_t
            - 10
            * np.log10(
                (p_tx_e * lambda_m**2 * g_theta_t_phi_t_f_t**2) / (16 * np.pi**2)
            )
        )

        return TS_m

    #
    # Chapter IV: VOLUME BACKSCATTERING STRENGTH
    #

    @staticmethod
    def calcSv(
        p_rx_e_n, r_c_n, lambda_f_c, p_tx_e, alpha_f_c, c, tau_eff, psi_f_c, g_0_f_c
    ):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        G = (p_tx_e * lambda_f_c**2 * c * tau_eff * psi_f_c * g_0_f_c**2) / (
            32 * np.pi**2
        )
        Sv_n = (
            10 * np.log10(p_rx_e_n)
            + 20 * np.log10(r_c_n)
            + 2 * alpha_f_c * r_c_n
            - 10 * np.log10(G)
        )

        return Sv_n

    @staticmethod
    def calcPulseCompSphericalSpread(y_pc_n, r_c_n):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        y_pc_s_n = y_pc_n * r_c_n

        return y_pc_s_n

    @staticmethod
    def defHanningWindow(c, tau, dr, f_s_dec):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        """
        Length of Hanning window currently chosen as 2^k samples for
        lowest k where 2^k >= 2 * No of samples in pulse
        """
        L = (c * 2 * tau) / dr  # Number of samples in 2 x pulse duration
        
        N_w = int(2 ** np.ceil(np.log2(L)))
        # or : N_w = np.ceil(2 ** np.log2(L)) - Length of Hanning window
        t_w_n = np.arange(0, N_w) / f_s_dec
        t_w = N_w / f_s_dec

        w_i = Calculation.hann(N_w)
        w_tilde_i = w_i / (np.linalg.norm(w_i) / np.sqrt(N_w))

        return w_tilde_i, N_w, t_w, t_w_n

    @staticmethod
    def calcDFTforSv(
        y_pc_s_n, w_tilde_i, y_mf_auto_n, N_w, n_f_points, f_m, f_s_dec, r_c_n, step
    ):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # Prepare for append
        Y_pc_v_m_n = []
        Y_tilde_pc_v_m_n = []
        svf_range = []

        # DFT of auto correlation function of the matched filter signal
        _Y_mf_auto_m = np.fft.fft(y_mf_auto_n, n=N_w)
        Y_mf_auto_m = Calculation.freqtransf(_Y_mf_auto_m, f_s_dec, f_m)

        min_sample = 0  # int(r0 / dr)
        max_sample = len(y_pc_s_n)  # int(r1 / dr)

        bin_start_sample = min_sample
        bin_stop_sample = bin_start_sample + N_w

        while bin_stop_sample < max_sample:
            # Windowed data
            yspread_bin = w_tilde_i * y_pc_s_n[bin_start_sample:bin_stop_sample]

            # Range for bin
            bin_center_sample = int((bin_stop_sample + bin_start_sample) / 2)
            bin_center_range = r_c_n[bin_center_sample]
            svf_range.append(bin_center_range)

            # DFT of windowed data
            _Y_pc_v_m = np.fft.fft(yspread_bin, n=N_w)
            Y_pc_v_m = Calculation.freqtransf(_Y_pc_v_m, f_s_dec, f_m)

            # Normalized DFT of windowed data
            Y_tilde_pc_v_m = Y_pc_v_m / Y_mf_auto_m

            # Append data
            Y_pc_v_m_n.append([Y_pc_v_m])
            Y_tilde_pc_v_m_n.append([Y_tilde_pc_v_m])

            # Next range bin
            bin_start_sample += step
            bin_stop_sample = bin_start_sample + N_w

        svf_range = np.array(svf_range)

        return Y_pc_v_m_n, Y_mf_auto_m, Y_tilde_pc_v_m_n, svf_range

    @staticmethod
    def freqtransf(FFTvecin, fsdec, fvec=None):
        """
        Shift FFT frequencies for specified frequencies.
        
        Parameters
        ----------
        FFTvecin : np.array
            FFT data from decimated frequencies
        fsdec : float
            Decimated sampling frequency [Hz]
        fvec : np.array
            Specified frequencies [Hz]
            
            From calibration data. ("CAL": {"frequencies")
                      If no calibration generate freq vector starting from 
                       f0 to f1 with same number of points as in calibration data
                    
        Returns
        -------
        float
            Vector with corrected frequencies [Hz]
        """

        nfft = len(FFTvecin)
        idxtmp = np.floor(fvec / fsdec * nfft).astype("int")
        idx = np.mod(idxtmp, nfft)

        return FFTvecin[idx]

    @staticmethod
    def calcPowerFreqSv(Y_tilde_pc_v_m_n, N_u, z_rx_e, z_td_e):
        """
        XXX.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # Initialize list of power values by range
        P_rx_e_v_m_n = []

        # Impedances
        Z = (np.abs(z_rx_e + z_td_e) / np.abs(z_rx_e)) ** 2 / np.abs(z_td_e)

        # Loop over list of FFTs along range
        for Y_tilde_pc_v_m in Y_tilde_pc_v_m_n:
            P_rx_e_v_m = N_u * (np.abs(Y_tilde_pc_v_m) / (2 * np.sqrt(2))) ** 2 * Z
            # Append power to list
            P_rx_e_v_m_n.append(P_rx_e_v_m)

        return P_rx_e_v_m_n

    @staticmethod
    def calcSvf(P_rx_e_t_m_n, alpha_m, p_tx_e, lambda_m, t_w,
                psi_m, g_0_m, c, svf_range):
        """
        Calculate Sv as a function of frequency.
        
        Parameters
        ----------
        P_rx_e_t_m_n : np.array
            DFT of the received electric power [W]
        alpha_m : float
            Acoustic absorption [dB/km]
        p_tx_e : float
            Transmitted electric power [W]
        lambda_m : float
            Acoustic wavelength [m]
        t_w : float
            Sliding window duration [s]
        psi_m : float
            Equivalent beam angle [sr]
        g_0_m : float
            Transducer gain [dB]
        c : float
            Speed of sound [m/s]
        svf_range : np.array
            XXX [m]
            
        Returns
        -------
        np.array
            Sv(f) [dB re 1 m^-1]
        """
        
        # Initialize list of Svf by range
        Sv_m_n = np.empty([len(svf_range), len(alpha_m)], dtype=float)

        G = (p_tx_e * lambda_m**2 * c * t_w * psi_m * g_0_m**2) / (32 * np.pi**2)
        n = 0

        # Loop over list of power values along range
        for P_rx_e_t_m in P_rx_e_t_m_n:
            Sv_m = (
                10 * np.log10(P_rx_e_t_m)
                + 2 * alpha_m * svf_range[n]
                - 10 * np.log10(G)
            )

            # Add to array
            Sv_m_n[n, ] = Sv_m
            n += 1

        return Sv_m_n

    @staticmethod
    def calcpsi(psi_f_n, f_n, f_m):
        """
        Calculate psi at given frequency.
        
        Parameters
        ----------
        psi_f_n : float
            Psi at nominal frequency [sr]
        f_n : float
            Nominal frequency [Hz]
        f_m : float
            Frequency to calculate psi at [Hz]
            
        Returns
        -------
        float 
            Psi at frequency `f_m` [sr]
        """
        
        return psi_f_n * (f_n / f_m) ** 2
