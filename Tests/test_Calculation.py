import unittest
import numpy as np
import scipy.io

from Core.Calculation import Calculation


class TestCalculation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.calc = Calculation(r'..\Data\pyEcholabEK80data.json')


    def test_hann(self):
        # Calculation trough implemented methods
        w = Calculation.hann(7)

        # Ground truth
        gt_w = np.array([0., 0.25, 0.75, 1., 0.75, 0.25, 0.])

        # Test
        np.testing.assert_allclose(w, gt_w, rtol=0, atol=1e-15)

    def test_normalized_hann(self):
        c, tau, dr, f_s_dec = 1482.0, 0.002047999994829297, 0.007904024915660557, 93750.0

        w_tilde_i_true = np.array([
            0.00000000e+00, 1.54079067e-05, 6.16310456e-05, 1.38667673e-04,
            2.46514883e-04, 3.85168607e-04, 5.54623615e-04, 7.54873514e-04,
            9.85910750e-04, 1.24772661e-03
        ])
        N_w_true = 1024
        t_w_true = 0.010922666666666667
        t_w_n_true = np.array([
            0.00000000e+00, 1.06666667e-05, 2.13333333e-05, 3.20000000e-05,
            4.26666667e-05, 5.33333333e-05, 6.40000000e-05, 7.46666667e-05,
            8.53333333e-05, 9.60000000e-05
        ])

        w_tilde_i, N_w, t_w, t_w_n = Calculation.defHanningWindow(c, tau, dr, f_s_dec)

        np.testing.assert_equal(N_w, N_w_true)
        np.testing.assert_almost_equal(t_w, t_w_true, decimal=6)
        np.testing.assert_allclose(w_tilde_i[0:10], w_tilde_i_true, rtol=0, atol=1e-10)
        np.testing.assert_allclose(t_w_n[0:10], t_w_n_true, rtol=0, atol=1e-10)



    def test_chirp(self):
        # Calculation trough implemented methods
        signal = Calculation.chirp(np.arange(0, 5), 1, 5, 2)

        # Ground truth
        gt_signal = np.array([1., 0.80901699, -0.80901699, 0.80901699, -0.80901699])

        # Test
        np.testing.assert_allclose(signal, gt_signal, rtol=0, atol=1e-8)

    def test_generateIdealWindowedTransmitSignal(self):
        f_0, f_1, tau, f_s, slope = 90000.0, 170000.0, 0.002047999994829297, 5000, 0.5
        y_tx_n, t = Calculation.generateIdealWindowedTransmitSignal(f_0, f_1, tau, f_s, slope)

        y_tx_n_true = np.array([
            0., 0.02282123, 0.29215947, 0.73558894, -0.96984631,
            -0.95121093, 0.53032985, -0.08060687, 0.11697778, 0.
        ])

        t_true = np.array([0.    , 0.0002, 0.0004, 0.0006, 0.0008, 0.001 , 0.0012, 0.0014, 0.0016, 0.0018])

        np.testing.assert_allclose(t, t_true, rtol=0, atol=0.0001)
        np.testing.assert_allclose(y_tx_n, y_tx_n_true, rtol=0, atol=0.0001)



    def test_calcAvgSumQuad(self):
        # Calculation trough implemented methods
        ycq = np.array([[1.0 + 10.0j, -1.0 - 10.0j],
                        [2.0 + 20.0j, -2.0 - 20.0j],
                        [3.0 + 30.0j, -3.0 - 30.0j],
                        [4.0 + 40.0j, -4.0 - 40.0j]])
        yc = Calculation.calcAverageSignal(ycq)

        # Ground truth
        gt_yc = np.array([2.5+25.j, -2.5-25.j])

        # Test
        np.testing.assert_allclose(yc, gt_yc, rtol=0, atol=0.0001)



    """
    def test_calculateTVGdB(self):

        # Calculation trough implemented methods
        alpha = 0.010425962012642035
        r = np.array([1, 50, 100, 500])
        tvg = Calculation.calculateTVGdB(alpha, r)

        # Ground truth
        gt_tvg = np.array([2.08519240e-02, 3.50219963e+01, 4.20851924e+01, 6.44053621e+01])

        # Test
        np.testing.assert_allclose(tvg, gt_tvg, rtol=0, atol=1e-6)
    """

    def test_calcPower(self):
        # Calculation trough implemented methods
        z_td_e, z_rx_e, N_u = 75, 5400, 4
        y_pc_n = np.array(
            [-9.99537661e-05 + 2.97907232e-05j, -2.63518109e-05 - 9.63090422e-05j, 6.27873532e-05 - 3.33523700e-07j,
             -2.04267017e-05 + 3.20007899e-05j, -2.94661296e-05 - 1.66837242e-05j])
        power = self.calc.calcPower(y_pc_n, z_td_e, z_rx_e, N_u)

        # Ground truth
        gt_power = np.array([8.3808e-11, 7.6809e-11, 3.0373e-11, 1.1104e-11, 8.8336e-12])

        # Test
        np.testing.assert_allclose(power, gt_power, rtol=0, atol=1e-9)

    def test_alignAuto_auto_gt_yc(self):

        auto = np.array([-0.7-0.7j, -0.6-0.6j, -0.7-0.7j, -0.8-0.8j, -0.9-0.9j, 1+1j, 0.9+0.9j, 0.8+0.8j, 0.7+0.7j, 0.6+0.6j, 0.5+0.5j])
        yc = np.array([-0.8-0.8j, -0.9-0.9j, 1+1j, 0.9+0.9j, 0.8+0.8j, 0.7+0.7j])

        _auto = Calculation.alignAuto(auto, yc)

        np.testing.assert_equal(_auto, np.array([
            -0.8-0.8j, -0.9-0.9j,  1. +1.j ,  0.9+0.9j,  0.8+0.8j,  0.7+0.7j
        ]))

    def test_freqtransf(self):
        testData = scipy.io.loadmat('freqtransf.mat')

        FFTvec = Calculation.freqtransf(testData['FFTvecin'].squeeze(), testData['fsdec'].squeeze(),
                                        testData['fvec'].squeeze())

        np.testing.assert_allclose(FFTvec, testData['FFTvec'].squeeze(), rtol=0.01, atol=0.1)

    def test_freqtransf2(self):

        FFTvecin = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j, 6+6j, 7+7j, 8+8j, 9+9j, 10+10j])
        fsdec = 200
        fvec = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])

        FFTvec = Calculation.freqtransf(FFTvecin, fsdec, fvec)
        FFTvec_true = np.array([
            6. +6.j,  6. +6.j,  7. +7.j,  7. +7.j,  8. +8.j,  8. +8.j,9. +9.j,  9. +9.j, 10.+10.j, 10.+10.j,  1. +1.j
        ])
        np.testing.assert_allclose(FFTvec, FFTvec_true, rtol=0.01, atol=0.1)


    def test_pulseCompression(self):
        y_mf_n = np.array([
            0.00000000e+00 + 0.j, 8.18771469e-03 - 0.00582447j,
            -3.07992800e-02 - 0.01009615j, 1.02180327e+00 - 0.28663263j,
            -8.36105932e-01 - 0.58982137j, -1.00772648e+00 + 0.10053897j,
            -7.42972244e-01 - 0.69740986j, 9.89186451e-01 + 0.01900877j,
            -4.75021920e-02 + 0.04353109j, 1.74240029e-03 - 0.01309336j,
            7.81659333e-04 - 0.00363211j])

        y_rx_nu = np.array([
            [
                -1.17646799e-04 - 3.95109419e-05j, 3.74271971e-04 + 3.22387095e-05j,
                -1.29431006e-04 + 4.91670617e-05j, -3.69502115e-04 - 1.91425075e-04j,
                3.19951891e-06 + 1.16657147e-05j, 1.88612830e-05 + 3.48608628e-05j,
                -1.04294209e-04 - 2.47430318e-04j, -4.30864093e-05 - 1.34203830e-04j,
                3.34305478e-05 + 3.43852771e-05j, 6.49533977e-05 - 5.79879670e-05j
            ], [
                -1.98100970e-04 - 4.49943145e-05j, 2.85173319e-05 - 3.25633242e-04j,
                1.53754256e-04 + 2.15744061e-04j, -6.51946175e-05 + 2.57920794e-04j,
                9.49954338e-05 - 5.73917896e-05j, 8.54586251e-05 + 5.51276062e-05j,
                -1.31343622e-04 + 7.97366447e-05j, -2.17818870e-05 - 1.24406952e-05j,
                2.62514423e-05 + 3.48618087e-05j, -8.28368793e-05 + 3.86763531e-05j
            ], [
                -1.06678090e-04 - 6.42403102e-05j, 2.69366603e-04 - 2.55059596e-04j,
                -8.83403663e-06 + 1.95716493e-04j, -2.00008435e-04 + 1.60429816e-04j,
                1.88088030e-04 + 4.51141786e-05j, 1.03579870e-04 + 1.12163390e-04j,
                -1.69490071e-04 + 3.73643670e-05j, -4.52302629e-05 + 8.25982352e-05j,
                2.24370451e-05 + 1.21700556e-04j, -3.42654894e-05 + 2.67060682e-06j
            ], [
                5.72731769e-05 - 1.17646116e-04j, 2.59828870e-04 + 2.19559515e-04j,
                -3.00836167e-04 - 5.60939907e-06j, -2.70318822e-04 - 2.22896357e-04j,
                3.72458526e-05 + 9.02283864e-05j, 2.92327677e-05 - 4.24897917e-05j,
                -2.63714082e-05 - 3.40891158e-04j, -4.33252244e-05 - 1.59953968e-04j,
                6.87699358e-05 + 2.41661419e-05j, 1.25144259e-04 - 6.99625889e-05j
            ]
        ])

        yc_q = self.calc.calcPulseCompressedSignals(y_rx_nu, y_mf_n)

        y_rx_nu_true = np.array([[-2.59929093e-05 - 7.08515524e-05j, 4.01230017e-05 + 6.88343097e-05j,
                                  5.44954531e-05 + 5.00020181e-05j, 7.75840390e-06 - 2.80227782e-05j,
                                  -2.28696158e-05 - 1.90829994e-05j, 1.43009407e-06 + 2.53903893e-05j,
                                  1.57123588e-05 - 8.17875901e-06j, -2.56938627e-07 + 5.58841100e-07j,
                                  1.66534536e-07 - 1.84753072e-08j, 0.00000000e+00 + 0.00000000e+00j],
                                 [-4.81977983e-05 + 2.11785226e-05j, 3.95359007e-05 - 1.34791909e-05j,
                                  4.90301426e-06 - 2.27668426e-06j, -2.33582484e-05 - 1.60195143e-05j,
                                  5.70369567e-06 - 1.26589658e-05j, 1.19765470e-05 - 7.27732686e-06j,
                                  -1.85759411e-05 + 2.82262001e-06j, 4.16106349e-07 - 3.04353480e-07j,
                                  -1.73035461e-07 - 3.17549098e-08j, 0.00000000e+00 + 0.00000000e+00j],
                                 [-8.89594135e-05 - 3.99596921e-06j, 3.95732254e-05 + 1.33507323e-05j,
                                  1.97328309e-05 - 2.61265759e-05j, -3.53574024e-05 - 4.93843120e-05j,
                                  -2.30959308e-05 - 3.38454569e-06j, 2.69480411e-06 + 2.00438241e-05j,
                                  -7.38271558e-06 - 1.95376433e-06j, 9.63805888e-08 + 1.33854988e-07j,
                                  -5.67093483e-08 - 3.40343413e-08j, 0.00000000e+00 + 0.00000000e+00j],
                                 [-2.10448797e-05 - 4.66392941e-05j, 4.14630362e-05 + 1.17446448e-04j,
                                  6.59347296e-05 + 6.75266508e-05j, 1.71564228e-05 - 2.62658612e-05j,
                                  -3.80306135e-05 - 1.67625934e-05j, 8.98775918e-07 + 3.41388745e-05j,
                                  2.79880739e-05 - 7.12997829e-06j, -5.22006796e-07 + 7.69251172e-07j,
                                  2.74274767e-07 + 2.98886629e-08j, 0.00000000e+00 + 0.00000000e+00j]])

        np.testing.assert_allclose(yc_q[0][0:4], y_rx_nu_true[0][0:4], rtol=0.01, atol=0.1)

    """
    def test_C1Prx(self):
        np.testing.assert_almost_equal(self.calc.C1Prx, 0.0068531, 5)
    """

    def test_Gf(self):
        Gf = self.calc.calcg0(39500)

        np.testing.assert_almost_equal(Gf, 26.11500, 5)

    def test_PSIfc(self):
        f_0, f_1, n_f_points = 90000.0, 170000.0, 10

        f_m = np.linspace(f_0, f_1, n_f_points)
        psifc = self.calc.calcpsi(0.008511380382023767, 38000.0, f_m)

        ans = [0.00151734, 0.00125682, 0.00105806, 0.00090297, 0.00077964, 0.00067996, 0.00059824, 0.00053041,
               0.0004735, 0.00042527]

        np.testing.assert_almost_equal(psifc, ans, 5)

    """
    def test_logSvCf(self):

        logSvCf = self.calc.calculateCSvfdB(39500)

        np.testing.assert_almost_equal(logSvCf, 11.147621416454323,5)
    """

    """
    def test_range(self):

        range,_ = self.calc.calcRange()
        np.testing.assert_almost_equal(range[0:4], [0.03146669, 0.06293338, 0.09440007, 0.12586676],1)
    """

    """
    def test_taueff(self):
        np.testing.assert_almost_equal(self.calc.tau_eff, 9.986813363835193e-05, 5)

    """
    """
    def test_Svf(self):
        ycq = self.calc.calcPulseCompressedSignals(self.calc.y_rx_org)
        yc = self.calc.calcAverageSignal(ycq)
        Svf, svf_range, f = self.calc.calcSvf(yc,10,30)

        np.testing.assert_allclose(Svf[0,:][0:4], [-58.21107559, -58.23425062, -58.25742747, -58.28060613], rtol=0.0001, atol=0.001)
        np.testing.assert_allclose(svf_range,[14.03414407, 18.06188048, 22.0896169 , 26.05441993], rtol=0.0001, atol=0.001)
        np.testing.assert_allclose(f[0:4], [34000., 34011.01101101, 34022.02202202, 34033.03303303], rtol=0.0001,
                                   atol=0.001)
    """


if __name__ == '__main__':
    unittest.main()
