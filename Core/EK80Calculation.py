import numpy as np

from Core.EK80DataContainer import EK80DataContainer


class EK80Calculation(EK80DataContainer):

    def __init__(self, jsonfname=None):
        super().__init__(jsonfname)

        # Derived variables
        """
        if self.hasData is not None:
            self.ekdata = ekdata
            self.z_rx_e = ekdata.z_rx_e  # (Ohm) WBT receiving impedance
            self.ycq = ekdata.quadrant_signals
            self.ycq_pc = ekdata.ycq_pc
            self.yc = ekdata.yc
            self.power = ekdata.yc
            self.Sv = ekdata.Sv
            self.Svf = ekdata.Svf
        """

        # Constants

        self.z_trd = 75  # (Ohm) Transducer impedance
        self.f_s = 1.5e6  # (Hz) Orginal WBT sampling rate
        self.n_f_points = 1000  # Number of frequency points for evaluation of TS(f) and Sv(f)

        # Derived constants
        # Constant used to calculate power, some factors can be simplified, but this is written for clarity
        K1 = 4 / ((2 * np.sqrt(2)) ** 2)
        K2 = (np.abs(self.z_rx_e + self.z_trd) / self.z_rx_e) ** 2
        K3 = 1.0 / np.abs(self.z_trd)
        self.C1Prx = K1 * K2 * K3
        self.f_s_dec = 1
        self.calculateDerivedVariables()

    def calculateDerivedVariables(self):

        # Estmate center frequency for transmit pulse
        self.f_c = (self.f0 + self.f1) / 2.0

        # Generate ideal transmit pulse at original sampling rate
        y_tx, t = EK80Calculation.generateIdealWindowedSendPulse(self.f0, self.f1, self.tau, self.f_s,
                                                                 self.slope)
        self.ytx0 = y_tx
        self.ytx0_t = t

        # Normalize ideal transmit pulse
        y_tx = y_tx / np.max(y_tx)

        # Filter and decimate ideal transmit pulse trough stage filters and calculate decimated sampling rate
        self.f_s_dec *= self.f_s
        if self.fil1s is not None:
            for fil1 in self.fil1s:
                y_tx = self.stageFilter(y_tx, fil1)
                self.f_s_dec *= 1 / fil1.DecimationFactor

        # Output signal from the final filter and decimation stage is used as matched filter
        y_mf = y_tx
        self.y_mf = y_mf

        # Create complex conjugated time reversed version and 2-norm of matched filter
        y_mf_n = np.conj(y_mf)[::-1]
        y_mf_twoNormSquared = np.linalg.norm(y_mf, 2) ** 2
        self.y_mf_n = y_mf_n
        self.y_mf_twoNormSquared = y_mf_twoNormSquared

        # Calculate auto correlation function for matched filter
        y_mf_auto = np.convolve(y_mf, y_mf_n) / y_mf_twoNormSquared
        self.y_mf_auto = y_mf_auto

        # Estimate effective pulse duration
        p_tx_auto = np.abs(y_mf_auto) ** 2
        self.tau_eff = np.sum(p_tx_auto) / ((np.max(p_tx_auto)) * self.f_s_dec)

        # Calibration data
        if self.frequencies is not None:
            self.frequencies = np.array(self.frequencies)
        else:
            # If no calibration make a frequency vector
            # This is used to calculate correct frequencies after signal decimation
            noFreq = 112
            self.frequencies = np.linspace(self.f0, self.f1, noFreq)


    def calcPulseCompressedQuadrants(self, quadrant_signals):

        # Do pulse compression on all quadrants
        pulseCompressedQuadrants = []
        start_idx = len(self.y_mf_n) - 1
        for u in quadrant_signals:
            y_pc = np.convolve(self.y_mf_n, u, mode='full') / self.y_mf_twoNormSquared
            y_pc = y_pc[start_idx::] # Correct sample indexes for mached filter
            pulseCompressedQuadrants.append(y_pc)

        return np.array(pulseCompressedQuadrants)

    @staticmethod
    def calcAvgSumQuad(y_pc):
        return np.sum(y_pc, axis=0) / y_pc.shape[0]

    def calcPower(self, y_pc):
        return self.C1Prx * np.abs(y_pc) ** 2

    def calcRange(self):
        dr = self.sampleInterval * self.c * 0.5
        r = np.array([(self.offset + i + 1) * dr for i in range(0, self.sampleCount)])
        return r, dr

    def calcSv(self, power, r0=None, r1=None):

        Gfc = self.G_f(self.f_c)
        PSIfc = self.PSI_f(self.f_c)
        logSvCf = self.calculateCSvfdB(self.f_c)
        r, _ = self.calcRange()

        alpha_fc = self.calcAbsorption(self.temperature, self.salinity, self.depth, self.acidity, self.c, self.f_c)

        if r0 is not None and r1 is not None:
            Idx = np.where((r>=r0) & (r<=r1))
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



    def calcSvf(self, y_pc, r0, r1, overlap=0.5):

        r, dr = self.calcRange()
        yspread = y_pc * r

        """
            Length of Hanning window currently chosen as 2^k samples for lowest k where 2^k >= 2 * No of samples in pulse
        """
        L = ((self.c * 2 * self.tau) / dr) # Number of samples in pulse duration

        Nw = int(2 ** np.ceil(np.log2(L))) # or : Nw = np.ceil(2 ** np.log2(L)) - Length of Hanning window
        tw = Nw / self.f_s_dec

        w = EK80Calculation.hann(Nw)
        w = w / (np.linalg.norm(w) / np.sqrt(Nw))

        step = int(Nw * (1 - overlap))

        f = np.linspace(self.f0, self.f1, self.n_f_points)

        _FFTytxauto = np.fft.fft(self.y_mf_auto, n=Nw)
        FFTytxauto = self.freqtransf(_FFTytxauto, self.f_s_dec, f)

        Gf = self.G_f(f)    # Used only if not calibrated
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
        while not last_bin :
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
                w = EK80Calculation.hann(len(sub_yspread))
                w = w / (np.linalg.norm(w) / np.sqrt(len(sub_yspread)))
                yspread_bin = w * sub_yspread

            bin_center_sample = int((bin_stop_sample+bin_start_sample) / 2)
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
    def alignAuto(auto, yc):

        idx_peak_auto = np.argmax(auto)
        idx_peak_yc = np.argmax(yc)

        left_samples_yc = idx_peak_yc
        right_samples_yc = len(yc) - idx_peak_yc

        idx_start_auto = max(0, idx_peak_auto - left_samples_yc)
        idx_stop_auto = min(len(auto), idx_peak_auto + right_samples_yc)

        new_auto = auto[idx_start_auto : idx_stop_auto]

        return new_auto

    def calcTSf(self, yc_target, peakIdx):

        """
        Usually the extracted target signal is shorter than the entire length of the
        auto correlation function of the transmit signal. In order to compensate correctly we need to extract the same samples from the auto correlation of the
        transmit signal as from the received signal relative to the peak signal sample.
        The reduced auto correlation signal of the transmit signal is labeled
        How to find samples from ytx_auto?

        These FFT’s represent the frequency band 0 − fs;dec with a resolution of fs;dec
        nff t and should be transferred to the actual frequency band through simple
        repetitions of the FFT’s and extraction of the original frequency band.
        How to extract the orginal frequency band?

        """
        if len(yc_target) <= len(self.y_mf_auto):


            # Middel of autocorelation function should allign with peakIdx
            ytx_auto_peakIdx = len(self.y_mf_auto) // 2

            reduced_ytx_auto = self.y_mf_auto[ytx_auto_peakIdx-peakIdx:ytx_auto_peakIdx+(len(yc_target)-peakIdx)]

        pass

    def PSI_f(self, f):
        return self.PSIfnom + 20 * np.log10(self.fnom / f)


    def G_f(self,f):
        if self.frequencies is None:
            # Uncalibrated case
            return self.Gfnom + 20 * np.log10(f / self.fnom)
        else:
            # Calibrated case
            return np.interp(f, self.frequencies, self.gain)


    def lambda_f(self, f):
        return self.c / f

    def calcElectricalAngles(self, y_pc):

        # Transduceres might have different segment configuration
        # Here we assume 4 quadrants

        y_pc_fore = 0.5 * (y_pc[2, :] + y_pc[2, :])
        y_pc_aft = 0.5 * (y_pc[0, :] + y_pc[1, :])
        y_pc_star = 0.5 * (y_pc[0, :] + y_pc[3, :])
        y_pc_port = 0.5 * (y_pc[1, :] + y_pc[2, :])

        y_alon = np.arctan2(np.real(y_pc_fore), np.imag(np.conj(y_pc_aft))) * 180 / np.pi
        y_athw = np.arctan2(np.real(y_pc_star), np.imag(np.conj(y_pc_port))) * 180 / np.pi

        return y_alon, y_athw

    @staticmethod
    def generateIdealWindowedSendPulse(f0, f1, tau, fs, slope):
        nsamples = int(np.floor(tau * fs))
        t = np.linspace(0, nsamples-1, num=nsamples) * 1/fs
        y = EK80Calculation.chirp(t, f0, tau, f1)
        L = int(np.round(tau * fs * slope * 2.0))  # Length of hanning window
        w = EK80Calculation.hann(L)
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
        # signal = signal / np.max(np.abs(signal))
        return np.convolve(signal, filter.Coefficients, mode='full')[0::filter.DecimationFactor]

    @staticmethod
    def calculateTVGdB(alpha, r):
        return 20.0 * np.log10(r) + 2.0 * alpha * r

    def calculateCSvfdB(self, f):
        lf = self.lambda_f(f)
        return 10 * np.log10((self.ptx * self.c * lf ** 2) / (32.0 * np.pi * np.pi))

    @staticmethod
    def calcAbsorption(t, s, d, ph, c, f):
        f = f / 1000

        a1 = (8.86 / c) * 10 ** (0.78 * ph - 5)
        p1 = 1
        f1 = 2.8 * (s / 35) ** 0.5 * 10 ** (4 - 1245 / (t + 273))

        a2 = 21.44 * (s / c) * (1 + 0.025 * t)
        p2 = 1 - 1.37e-4 * d + 6.62e-9 * d ** 2
        f2 = 8.17 * 10 ** (8 - 1990 / (t + 273)) / (1 + 0.0018 * (s - 35))

        p3 = 1 - 3.83e-5 * d + 4.9e-10 * d ** 2

        a3l = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t ** 2 - 1.5e-8 * t ** 3
        a3h = 3.964e-4 - 1.146e-5 * t + 1.45e-7 * t ** 2 - 6.5e-10 * t ** 3
        a3 = a3l * (t <= 20) + a3h * (t > 20)

        a = f ** 2 * (a1 * p1 * f1 / (f1 ** 2 + f ** 2) + a2 * p2 * f2 / (f2 ** 2 + f ** 2) + a3 * p3)

        return a / 1000

