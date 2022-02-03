import unittest
import numpy as np
import scipy.io

from Core.EK80Calculation import EK80CalculationPaper


class TestEK80Calculation(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.calc = EK80CalculationPaper('..\Data\pyEcholabEK80data.json')


    def test_calcAvgSumQuad(self):

        # Calculation trough implemented methods
        ycq = np.array([[4.85690181+11.37198657j, -5.3561974 + 0.20880486j], [5.64515755+11.23851476j, -5.28595187 +0.55263475j], [ 5.40798186+11.21237538j, -5.30801666 +0.45015535j], [5.17319057+11.08695032j, -5.31563328 +0.38322727j]])
        yc = EK80CalculationPaper.calcAvgSumQuad(ycq)

        # Ground truth
        gt_yc = np.array([5.27081+11.2275j, -5.31645+0.398706j])

        # Test
        np.testing.assert_allclose(yc, gt_yc, rtol=0, atol=0.0001)

    def test_hann(self):

        # Calculation trough implemented methods
        w = EK80CalculationPaper.hann(7)

        # Ground truth
        gt_w = np.array([0., 0.25, 0.75, 1., 0.75, 0.25, 0.])

        # Test
        np.testing.assert_allclose(w, gt_w, rtol=0, atol=1e-15)

    def test_chirp(self):
        # Calculation trough implemented methods
        signal = EK80CalculationPaper.chirp(np.arange(0, 5), 1, 5, 2)

        # Ground truth
        gt_signal = np.array([1.,  0.80901699, -0.80901699,  0.80901699, -0.80901699])

        # Test
        np.testing.assert_allclose(signal, gt_signal, rtol=0, atol=1e-8)

    def test_calculateTVGdB(self):

        # Calculation trough implemented methods
        alpha = 0.010425962012642035
        r = np.array([1, 50, 100, 500])
        tvg = EK80CalculationPaper.calculateTVGdB(alpha, r)

        # Ground truth
        gt_tvg = np.array([2.08519240e-02, 3.50219963e+01, 4.20851924e+01, 6.44053621e+01])

        # Test
        np.testing.assert_allclose(tvg, gt_tvg, rtol=0, atol=1e-6)

    def test_calcPower(self):

        # Calculation trough implemented methods
        yc = np.array([-9.99537661e-05+2.97907232e-05j, -2.63518109e-05-9.63090422e-05j, 6.27873532e-05-3.33523700e-07j, -2.04267017e-05+3.20007899e-05j, -2.94661296e-05-1.66837242e-05j])
        power = self.calc.calcPower(yc)

        # Ground truth
        gt_power = np.array([8.3808e-11, 7.6809e-11, 3.0373e-11, 1.1104e-11, 8.8336e-12])

        # Test
        np.testing.assert_allclose(power, gt_power, rtol=0, atol=1e-9)

    def test_alignAuto_auto_gt_yc(self):

        auto = np.array([-0.7, -0.6, -0.7, -0.8, -0.9, 1, 0.9, 0.8, 0.7, 0.6, 0.5])
        yc = np.array([-0.8, -0.9, 1, 0.9, 0.8, 0.7])

        _auto = EK80CalculationPaper.alignAuto(auto, yc)

        np.testing.assert_equal(_auto, np.array([-0.8, -0.9, 1, 0.9, 0.8, 0.7]))

    def test_freqtransf(self):

        testData = scipy.io.loadmat('freqtransf.mat')

        FFTvec = EK80CalculationPaper.freqtransf(testData['FFTvecin'].squeeze(), testData['fsdec'].squeeze(), testData['fvec'].squeeze())

        np.testing.assert_allclose(FFTvec, testData['FFTvec'].squeeze(), rtol=0.01, atol=0.1)

    def test_pulseCompression(self):
        yc_q = self.calc.calcPulseCompressedQuadrants(self.calc.y_rx_org)

        yc_q_true = np.array([[-2.95021109e+01-89.58134313j,  2.07277970e+01-31.69831125j,
        -6.17307936e+01-20.10871076j, -3.68049196e-01+18.74086737j],
       [-2.90332009e+01-88.61826691j,  2.10273307e+01-31.62740527j,
        -6.17061094e+01-20.20206632j, -4.84996624e-02+19.03046629j],
       [-3.05403181e+01-90.29428665j,  2.14362707e+01-32.01943839j,
        -6.28027305e+01-20.50513911j, -1.82754919e-01+19.30941347j],
       [-3.00023000e+01-87.72991092j,  2.17024782e+01-32.98965848j,
        -6.29452825e+01-19.86017115j,  1.64285924e-01+19.18403556j]])

        np.testing.assert_allclose(yc_q[0][0:4],yc_q_true[0][0:4] , rtol=0.01, atol=0.1)


    def test_C1Prx(self):
        np.testing.assert_almost_equal(self.calc.C1Prx, 0.0068531, 5)

    def test_Gf(self):

        Gf = self.calc.calc_G0_m(39500)

        np.testing.assert_almost_equal(Gf,26.11500, 5)


    def test_PSIfc(self):

        PSIfc = self.calc.PSI_f(39500)

        np.testing.assert_almost_equal(PSIfc, -21.03626998, 5)

    def test_logSvCf(self):

        logSvCf = self.calc.calculateCSvfdB(39500)

        np.testing.assert_almost_equal(logSvCf, 11.147621416454323,5)

    def test_range(self):

        range,_ = self.calc.calcRange()
        np.testing.assert_almost_equal(range[0:4], [0.03146669, 0.06293338, 0.09440007, 0.12586676],1)


    def test_taueff(self):
        np.testing.assert_almost_equal(self.calc.tau_eff,9.986813363835193e-05,5)

    def test_Svf(self):
        ycq = self.calc.calcPulseCompressedQuadrants(self.calc.y_rx_org)
        yc = self.calc.calcAvgSumQuad(ycq)
        Svf, svf_range, f = self.calc.calcSvf(yc,10,30)

        np.testing.assert_allclose(Svf[0,:][0:4], [-58.21107559, -58.23425062, -58.25742747, -58.28060613], rtol=0.0001, atol=0.001)
        np.testing.assert_allclose(svf_range,[14.03414407, 18.06188048, 22.0896169 , 26.05441993], rtol=0.0001, atol=0.001)
        np.testing.assert_allclose(f[0:4], [34000., 34011.01101101, 34022.02202202, 34033.03303303], rtol=0.0001,
                                   atol=0.001)


if __name__ == '__main__':
    unittest.main()
