import numpy as np

from Core.EK80DataContainer import EK80DataContainer


class EK80CalculationPaper(EK80DataContainer):
            
    @staticmethod
    def calcAutoCorrelation(y_mf_n, f_s_dec):
        y_mf_n_conj_rev = np.conj(y_mf_n)[::-1]
        y_mf_twoNormSquared = np.linalg.norm(y_mf_n, 2) ** 2
        y_mf_n_conj_rev = y_mf_n_conj_rev
        y_mf_twoNormSquared = y_mf_twoNormSquared

        # Calculate auto correlation function for matched filter
        y_mf_auto_n = np.convolve(y_mf_n, y_mf_n_conj_rev)/y_mf_twoNormSquared

        # Estimate effective pulse duration
        p_tx_auto = np.abs(y_mf_auto_n) ** 2
        tau_eff = np.sum(p_tx_auto) / ((np.max(p_tx_auto)) * f_s_dec)
        return y_mf_auto_n, tau_eff

    @staticmethod
    def calcDecmiatedSamplingRate(filter_v, f_s):
        f_s_dec = [f_s]
        v = 0
        if filter_v is not None:
            for filter_v in filter_v:
                tmp = f_s_dec[v] / filter_v["D"]
                f_s_dec.append(tmp)
                v += 1
        return f_s_dec
    
    @staticmethod
    def calcNormalizedTransmitSignal(y_tx_n):
        return y_tx_n / np.max(y_tx_n)

    @staticmethod
    def calcFilteredAndDecimatedSignal(y_tilde_tx_n, filter_v):
        # Initialize with normalized transmit pulse
        y_tilde_tx_nv = [y_tilde_tx_n]
        v = 0
        if filter_v is not None:
            for filter_vi in filter_v:
                tmp = np.convolve(y_tilde_tx_nv[v],
                                  filter_vi["h_fl_i"],
                                  mode='full')[0::filter_vi["D"]]
                y_tilde_tx_nv.append(tmp)
                v += 1

        return y_tilde_tx_nv

    @staticmethod
    def calcPulseCompressedQuadrants(quadrant_signals,y_mf_n):
        """
        Generate matched filtered signal for each quadrant

        Returns:
        np.array: y_pc_nu pulseCompressedQuadrants
        """
        # Do pulse compression on all quadrants

        y_mf_n_conj_rev = np.conj(y_mf_n)[::-1]
        y_mf_twoNormSquared = np.linalg.norm(y_mf_n, 2) ** 2

        pulseCompressedQuadrants = []
        start_idx = len(y_mf_n_conj_rev) - 1
        for u in quadrant_signals:
            # Please check that the order is ok and that
            # the use of y_mf_n_conj_rev is ok. I did this after a beer.
            y_pc_nu = np.convolve(y_mf_n_conj_rev, u,
                                  mode='full') / y_mf_twoNormSquared
            # Correct sample indexes for mached filter
            y_pc_nu = y_pc_nu[start_idx::]
            pulseCompressedQuadrants.append(y_pc_nu)

        return np.array(pulseCompressedQuadrants)

    @staticmethod
    def calcAvgSumQuad(y_pc_nu):
        """
        Calculate the mean signal over all transducer sectors

        Input:
        np_array: y_pc_nu

        Returns:
        np.array: y_pc_n
        """
        return np.sum(y_pc_nu, axis=0) / y_pc_nu.shape[0]
    
    @staticmethod
    def calcPower(y_pc, z_td_e, z_rx_e, N_u):
        K1 = 4 / ((2 * np.sqrt(2)) ** 2)
        K2 = (np.abs(z_rx_e + z_td_e) / z_rx_e) ** 2
        K3 = 1.0 / np.abs(z_td_e)
        C1Prx = K1 * K2 * K3
        
        return C1Prx * np.abs(y_pc) ** 2

    @staticmethod
    def calcSp(
            p_rx_e_n,
            r_n,
            alpha_f_c,
            p_tx_e,
            lambda_f_c,
            g_0_f_c,
            r0=None,
            r1=None):
        # Pick a range of the data
        if r0 is not None and r1 is not None:
            Idx = np.where((r_n >= r0) & (r_n <= r1))
            r_n = r_n[Idx]
            p_rx_e_n = p_rx_e_n[Idx]
        # Calculate Sp
        S_p_n = 10.0 * np.log10(p_rx_e_n) + 40.0 * np.log10(r_n) + \
            2.0 * alpha_f_c * r_n - \
            10 * np.log10((p_tx_e * lambda_f_c ** 2 * g_0_f_c ** 2)
                          / (16.0 * np.pi ** 2))

        return S_p_n

    def calcSv(self, power, r0=None, r1=None):

        Gfc = self.G_f(self.f_c)
        PSIfc = self.PSI_f(self.f_c)
        logSvCf = self.calculateCSvfdB(self.f_c)
        r, _ = self.calcRange()

        alpha_fc = self.calcAbsorption(self.temperature, self.salinity, self.depth, self.acidity, self.c, self.f_c)

        if r0 is not None and r1 is not None:
            Idx = np.where((r >= r0) & (r <= r1))
            r = r[Idx]
            power = power[Idx]

        Sv = 10.0 * np.log10(power) + \
             20.0 * np.log10(r) + \
             2.0 * alpha_fc * r - \
             logSvCf - \
             2 * Gfc - \
             10 * np.log10(self.tau_eff) - \
             PSIfc

        return Sv, r

    @staticmethod
    def freqtransf(FFTvecin, fsdec, fvec=None):
        """
        Estimates fft data for Frequencies in fvec
        :param FFTvecin: fft data from decimated frequencies
        :param fsdec:   desimated sampling frequency. Decimation factors in FIL0 datagrams
        :param fvec:    Target frequencies. From calibration data. ("CAL": {"frequencies")
                        If no calibration - generate freq vector starting from f0 to f1 with same amount of points as in calibration data
        :return: Vector with corrected frequencies
        """

        nfft = len(FFTvecin)
        idxtmp = np.floor(fvec / fsdec * nfft).astype('int')
        idx = np.mod(idxtmp, nfft) + 1

        return FFTvecin[idx]

    def calcSvf_old(self, y_pc, r0, r1, overlap=0.5):

        r, dr = self.calcRange()
        yspread = y_pc * r

        """
            Length of Hanning window currently chosen as 2^k samples for lowest k where 2^k >= 2 * No of samples in pulse
        """
        L = ((self.c * 2 * self.tau) / dr)  # Number of samples in pulse duration

        Nw = int(2 ** np.ceil(np.log2(L)))  # or : Nw = np.ceil(2 ** np.log2(L)) - Length of Hanning window
        tw = Nw / self.f_s_dec

        w = EK80CalculationPaper.hann(Nw)
        w = w / (np.linalg.norm(w) / np.sqrt(Nw))

        step = int(Nw * (1 - overlap))

        f = np.linspace(self.f0, self.f1, self.n_f_points)

        _FFTytxauto = np.fft.fft(self.y_mf_auto_n, n=Nw)
        FFTytxauto = self.freqtransf(_FFTytxauto, self.f_s_dec, f)

        Gf = self.G_f(f)  # Used only if not calibrated
        PSIf = self.PSI_f(f)
        alpha_f = self.calcAbsorption(self.temperature, self.salinity, self.depth, self.acidity, self.c, f)
        logSvCf = self.calculateCSvfdB(f)
        svf_range = []
        Svf = []

        min_sample = int(r0 / dr)
        max_sample = int(r1 / dr)

        bin_start_sample = min_sample
        last_bin = False
        n_bins = 0
        while not last_bin:
            n_bins += 1
            bin_stop_sample = bin_start_sample + Nw

            # Apply window to signal bin
            if bin_stop_sample < max_sample:
                # We have a whole bin use precalculated window
                yspread_bin = w * yspread[bin_start_sample:bin_stop_sample]
            else:
                # We might have partial bin, recalculated window
                last_bin = True
                bin_stop_sample = max_sample
                sub_yspread = yspread[bin_start_sample:bin_stop_sample]
                w = EK80CalculationPaper.hann(len(sub_yspread))
                w = w / (np.linalg.norm(w) / np.sqrt(len(sub_yspread)))
                yspread_bin = w * sub_yspread

            bin_center_sample = int((bin_stop_sample + bin_start_sample) / 2)
            bin_center_range = r[bin_center_sample]
            svf_range.append(bin_center_range)

            _Fvolume = np.fft.fft(yspread_bin, n=Nw)
            Fvolume = self.freqtransf(_Fvolume, self.f_s_dec, f)

            FFTvolume_norm = Fvolume / FFTytxauto

            prx_FFT_volume = self.C1Prx * np.abs(FFTvolume_norm) ** 2

            _Svf = 10 * np.log10(prx_FFT_volume) + \
                   2 * alpha_f * bin_center_range - \
                   logSvCf - \
                   2 * Gf - \
                   10 * np.log10(tw) - \
                   PSIf

            Svf.append(_Svf)

            bin_start_sample += step

        Svf = np.array(Svf)
        svf_range = np.array(svf_range)

        return Svf, svf_range, f

    @staticmethod
    def alignAuto(y_mf_auto_n, y_pc_t_n):
        # The equivalent samples around the peak in the decimated tx signal
        idx_peak_auto = np.argmax(np.abs(y_mf_auto_n))
        idx_peak_y_pc_t_n = np.argmax(np.abs(y_pc_t_n))

        left_samples = idx_peak_y_pc_t_n
        right_samples = len(y_pc_t_n) - idx_peak_y_pc_t_n

        idx_start_auto = max(0, idx_peak_auto - left_samples)
        idx_stop_auto = min(len(y_mf_auto_n), idx_peak_auto + right_samples)

        y_mf_auto_red_n = y_mf_auto_n[idx_start_auto: idx_stop_auto]

        return y_mf_auto_red_n
    
    @staticmethod
    def singleTarget(y_pc_n, p_rx_e_n, theta_n, phi_n, r_n,
                     r0, r1, before=0.5, after=0.5):
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
    def calcDFTforTS(y_pc_t_n, y_mf_auto_red_n, n_f_points, f0, f1, f_s_dec):

        # The number of DFT points inpower of 2
        N_DFT = int(2 ** np.ceil(np.log2(n_f_points)))
        # Corresponding frequency vector
        
        f_m_t = np.linspace(f0, f1, n_f_points) # Ta ut denne og legg til f_m_t i argument
        # Y_pc_t_m = self.freqtransf(_Y_pc_t_m, self.f_s_dec, f_m)
        # def freqtransf(FFTvecin, fsdec, fvec=None):
        idxtmp = np.floor(f_m_t / f_s_dec * N_DFT).astype('int')
        idx = np.mod(idxtmp, N_DFT) + 1

        # DFT for the target signal
        _Y_pc_t_m = np.fft.fft(y_pc_t_n, n=N_DFT)
        Y_pc_t_m = _Y_pc_t_m[idx]

        # DFT for the transmit signal
        _Y_mf_auto_red_m = np.fft.fft(y_mf_auto_red_n, n=N_DFT)
        Y_mf_auto_red_m = _Y_mf_auto_red_m[idx]
        
        # The Normalized DFT
        Y_tilde_pc_t_m = Y_pc_t_m/Y_mf_auto_red_m
        
        return Y_pc_t_m, Y_mf_auto_red_m, Y_tilde_pc_t_m, f_m_t
    
    @staticmethod
    def calcPowerFreq(N_u, Y_tilde_pc_t_m, z_td_e, z_rx_e):
        imp = (np.abs(z_rx_e + z_td_e) / np.abs(z_rx_e)) ** 2 / np.abs(z_td_e)
        P_rx_e_t_m = N_u * (np.abs(Y_tilde_pc_t_m)/(2 * np.sqrt(2))) ** 2 * imp
        return P_rx_e_t_m

    @staticmethod
    def calcTSf(P_rx_e_t_m, r_t, alpha_m, p_tx_e,
                lambda_m, g_theta_t_phi_t_f_t):

        TS_m = 10*np.log10(P_rx_e_t_m) \
               + 40*np.log10(r_t) \
               + 2*alpha_m*r_t \
               - 10*np.log10((p_tx_e * lambda_m**2 * g_theta_t_phi_t_f_t ** 2) / (16 * np.pi ** 2))
        return TS_m
    
    """
        \begin{equation}
        \label{eq:TS_f}
        \ts(\samplesymf) = 10\log_{10}(\prxetf(\samplesymf)) + 40\log_{10}(\range) + 2\absorp(\samplesymf)\range 
        - 10\log_{10}\left( \frac{\ptxe \wlen_\samplesymf^2 \gain^2(\along,\athw,\samplesymf)}{16\pi^2} \right).
        \end{equation}

           logSpCf = self.calculateCSpfdB(f_m)

TS_m = 10 * np.log10(P_rx_e_t_m) + \
               40 * np.log10(r) + \
               2 * alpha_m * r - \
               2 * G_theta_phi_m - \
               logSpCf
        
        return TS_m

        """
    def calcTSf_old(self, r, theta, phi, y_pc_t_n):

        # L = len(y_pc_t_n)
        L = self.n_f_points
        N_DFT = int(2 ** np.ceil(np.log2(L)))  # or : Nw = np.ceil(2 ** np.log2(L))

        f_m = np.linspace(self.f0, self.f1, self.n_f_points)

        y_mf_auto_red_n = self.alignAuto(self.y_mf_auto_n, y_pc_t_n)

        _Y_pc_t_m = np.fft.fft(y_pc_t_n, n=N_DFT)
        Y_pc_t_m = self.freqtransf(_Y_pc_t_m, self.f_s_dec, f_m)

        _Y_mf_auto_red_m = np.fft.fft(y_mf_auto_red_n, n=N_DFT)
        Y_mf_auto_red_m = self.freqtransf(_Y_mf_auto_red_m, self.f_s_dec, f_m)

        G0_m = self.calc_g0_m(f_m)
        B_theta_phi_m = self.calc_b_theta_phi_m(theta, phi, f_m)
        G_theta_phi_m = G0_m - B_theta_phi_m
        alpha_m = self.calcAbsorption(self.temperature, self.salinity, self.depth, self.acidity, self.c, f_m)
        logSpCf = self.calculateCSpfdB(f_m)

        Y_tilde_pc_t_m = Y_pc_t_m / Y_mf_auto_red_m

        P_rx_e_t_m = self.C1Prx * np.abs(Y_tilde_pc_t_m) ** 2

        TS_m = 10 * np.log10(P_rx_e_t_m) + \
               40 * np.log10(r) + \
               2 * alpha_m * r - \
               2 * G_theta_phi_m - \
               logSpCf

        return TS_m, f_m

    def PSI_f(self, f):
        return self.PSI_fnom + 20 * np.log10(self.fnom / f)

    def G_f(self, f):
        if self.frequencies is None:
            # Uncalibrated case
            return self.Gfnom + 20 * np.log10(f / self.fnom)
        else:
            # Calibrated case
            return np.interp(f, self.frequencies, self.gain)

    def calc_g0_m(self, f):
        if self.isCalibrated:
            # Calibrated case
            return np.interp(f, self.frequencies, self.gain)
        else:
            # Uncalibrated case
            return self.G_fnom + 20 * np.log10(f / self.fnom)



    @staticmethod
    def calcTransducerHalves(y_pc_nu):
        y_pc_fore_n = 0.5 * (y_pc_nu[2, :] + y_pc_nu[3, :])
        y_pc_aft_n = 0.5 * (y_pc_nu[0, :] + y_pc_nu[1, :])
        y_pc_star_n = 0.5 * (y_pc_nu[0, :] + y_pc_nu[3, :])
        y_pc_port_n = 0.5 * (y_pc_nu[1, :] + y_pc_nu[2, :])

        return y_pc_fore_n, y_pc_aft_n, y_pc_star_n, y_pc_port_n

    @staticmethod
    def calcGamma(angle_sensitivity_fnom, f_c, fnom):
        return angle_sensitivity_fnom * (f_c / fnom)

    @staticmethod
    def calcAngles(y_pc_halves,gamma_theta, gamma_phi):
        # Transducers might have different segment configuration
        # Here we assume 4 quadrants
        y_pc_fore_n, y_pc_aft_n, y_pc_star_n, y_pc_port_n = y_pc_halves

        y_theta_n = y_pc_fore_n * np.conj(y_pc_aft_n)
        y_phi_n = y_pc_star_n * np.conj(y_pc_port_n)

        theta_n = np.arcsin(np.arctan2(np.imag(y_theta_n), np.real(y_theta_n)) / gamma_theta) * 180 / np.pi
        phi_n = np.arcsin(np.arctan2(np.imag(y_phi_n), np.real(y_phi_n)) / gamma_phi) * 180 / np.pi

        return theta_n, phi_n

    @staticmethod
    def generateIdealWindowedSendPulse(f0, f1, tau, fs, slope):
        nsamples = int(np.floor(tau * fs))
        t = np.linspace(0, nsamples - 1, num=nsamples) * 1 / fs
        y = EK80CalculationPaper.chirp(t, f0, tau, f1)
        L = int(np.round(tau * fs * slope * 2.0))  # Length of hanning window
        w = EK80CalculationPaper.hann(L)
        N = len(y)
        w1 = w[0:int(len(w) / 2)]
        w2 = w[int(len(w) / 2):-1]
        i0 = 0
        i1 = len(w1)

        i2 = N - len(w2)
        i3 = N

        y[i0:i1] = y[i0:i1] * w1
        y[i2:i3] = y[i2:i3] * w2

        return y, t

    @staticmethod
    def hann(L):
        n = np.arange(0, L, 1)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (L - 1)))

    @staticmethod
    def chirp(t, f0, t1, f1):
        a = np.pi * (f1 - f0) / t1
        b = 2 * np.pi * f0
        return np.cos(a * t * t + b * t)

    @staticmethod
    def stageFilter(signal, filter):
        return np.convolve(signal, filter.Coefficients, mode='full')[0::filter.DecimationFactor]

    @staticmethod
    def calculateTVGdB(alpha, r):
        return 20.0 * np.log10(r) + 2.0 * alpha * r

    @staticmethod
    def calc_Psi_f(Psi_f_n, f_n, f_m_t):
        return Psi_f_n * (f_n / f_m_t)**2

    @staticmethod
    def calc_Sv(p_rx_e_n, r_c_n, lambda_f_c,
                p_tx_e, alpha_f_c, c, tau_eff,
                Psi_f_c, g_0_f_c):
        G = (p_tx_e * lambda_f_c ** 2 * c * tau_eff * Psi_f_c *
             g_0_f_c ** 2) / (32 * np.pi ** 2)
        Sv_n = 10*np.log10(p_rx_e_n) + 20*np.log10(
            r_c_n) + 2 * alpha_f_c * r_c_n - 10*np.log10(G)
        return Sv_n

    @staticmethod
    def calc_PulseCompSphericalSpread(y_pc_n, r_c_n):
        y_pc_s_n = y_pc_n * r_c_n
        return y_pc_s_n

    @staticmethod
    def calcDFTforSv(y_pc_s_n, w_tilde_i, y_mf_auto_n, N_w,
                     n_f_points, f0, f1, f_s_dec, r_c_n, step):

        # The frequency vector for the windowed signal
        f = np.linspace(f0, f1, n_f_points)

        # Initialize the fft as list (and append for each depth bin)
        Y_pc_v_m_n = []
        Y_tilde_pc_v_m_n = []

        
        # The DFT of the ACf of the mathced filter signal
        _Y_mf_auto_m = np.fft.fft(y_mf_auto_n, n=N_w)
        Y_mf_auto_m = EK80CalculationPaper.freqtransf(_Y_mf_auto_m, f_s_dec, f)

        svf_range = []
        min_sample = 0  # int(r0 / dr)
        max_sample = len(y_pc_s_n)  # int(r1 / dr)

        bin_start_sample = min_sample
        last_bin = False
        n_bins = 0
        while not last_bin:
            bin_stop_sample = bin_start_sample + N_w

            # Apply window to signal bin
            if bin_stop_sample < max_sample:
                # We have a whole bin use precalculated window
                yspread_bin = w_tilde_i * y_pc_s_n[
                    bin_start_sample:bin_stop_sample]
            else:
                # We might have partial bin, recalculated window
                
                # TODO: We need to present this in the paper. Nils Olav is
                # sceptic to do this.Should we remove the edge cases? They will
                # be very different to the "center" cases as they may have much
                # fewer samples?
                last_bin = True
                bin_stop_sample = max_sample
                sub_yspread = y_pc_s_n[bin_start_sample:bin_stop_sample]
                w = EK80CalculationPaper.hann(len(sub_yspread))
                w = w / (np.linalg.norm(w) / np.sqrt(len(sub_yspread)))
                yspread_bin = w * sub_yspread

            # Find the range for this window
            bin_center_sample = int((bin_stop_sample + bin_start_sample) / 2)
            bin_center_range = r_c_n[bin_center_sample]
            svf_range.append(bin_center_range)

            # Calculate the dft of the windowed signal
            _Y_pc_v_m = np.fft.fft(yspread_bin, n=N_w)
            Y_pc_v_m = EK80CalculationPaper.freqtransf(_Y_pc_v_m, f_s_dec, f)

            # Scale the DFT with the acf of the mathed filter signal
            Y_tilde_pc_v_m = Y_pc_v_m / Y_mf_auto_m

            # TODO: Append to data structure for all ranges for this ping
            Y_pc_v_m_n.append([Y_pc_v_m])
            Y_tilde_pc_v_m_n.append([Y_tilde_pc_v_m])
            bin_start_sample += step

            # Next range bin
            n_bins += 1

        svf_range = np.array(svf_range)

        return Y_pc_v_m_n, Y_mf_auto_m, Y_tilde_pc_v_m_n, svf_range

    @staticmethod
    def calcPowerFreqforSv(Y_tilde_pc_v_m_n, N_u, z_rx_e, z_td_e):

        # Initialize list of power values by range
        P_rx_e_v_m_n = []

        # Impedances
        Z = (np.abs(z_rx_e + z_td_e)/np.abs(z_rx_e)) ** 2 / np.abs(z_td_e)
        
        # Loop over list of FFTs along range
        for Y_tilde_pc_v_m in Y_tilde_pc_v_m_n:
            P_rx_e_v_m = N_u * (
                np.abs(Y_tilde_pc_v_m) / (2 * np.sqrt(2))) ** 2 * Z
            # Append power to list
            P_rx_e_v_m_n.append(P_rx_e_v_m)
        
        return P_rx_e_v_m_n

    def calcSvf(P_rx_e_t_m_n, alpha_m, p_tx_e, lambda_m, t_w,
                Psi_f, g_0_m, c, svf_range):
        
        # Initialize list of Svf by range
        Sv_m_n = []

        G = (p_tx_e * lambda_m ** 2 * c * t_w * Psi_f * g_0_m ** 2)/(
            32 * np.pi ** 2)
        n = 0
        # Loop over list of power values along range
        for P_rx_e_t_m in P_rx_e_t_m_n:
            Sv_m = 10 * np.log10(
                P_rx_e_t_m) + 2 * alpha_m * svf_range[n] - 10 * np.log10(G)
            # Append power to list
            n += 1
            Sv_m_n.append(Sv_m)
        
        return Sv_m_n

    @staticmethod
    def defHanningWindow(c, tau, dr, f_s_dec):
        """
            Length of Hanning window currently chosen as 2^k samples for
            lowest k where 2^k >= 2 * No of samples in pulse
        """
        L = ((c * 2 * tau) / dr)  # Number of samples in 2 x pulse duration
        
        N_w = int(2 ** np.ceil(np.log2(L)))
        # or : N_w = np.ceil(2 ** np.log2(L)) - Length of Hanning window
        t_w_n = np.arange(0, N_w) * f_s_dec
        t_w = N_w * f_s_dec
        
        w_i = EK80CalculationPaper.hann(N_w)
        w_tilde_i = w_i / (np.linalg.norm(w_i) / np.sqrt(N_w))
        
        return w_tilde_i, N_w, t_w, t_w_n

    
    def calculateCSvfdB(self, f):
        lf = self.lambda_f(f)
        return 10 * np.log10((self.ptx * self.c * lf ** 2) / (32.0 * np.pi * np.pi))

