import unittest
import numpy as np
import scipy.io

from Core.Calculation import Calculation


class TestCalculation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.calc = Calculation(r"..\Data\pyEcholabEK80data.json")

    def test_hann(self):
        # Calculation trough implemented methods
        w = Calculation.hann(7)

        # Ground truth
        gt_w = np.array([0.0, 0.25, 0.75, 1.0, 0.75, 0.25, 0.0])

        # Test
        np.testing.assert_allclose(w, gt_w, rtol=0, atol=1e-15)

    def test_normalized_hann(self):
        c, tau, dr, f_s_dec = (
            1482.0,
            0.002047999994829297,
            0.007904024915660557,
            93750.0,
        )

        w_tilde_i_true = np.array(
            [
                0.00000000e00,
                1.54079067e-05,
                6.16310456e-05,
                1.38667673e-04,
                2.46514883e-04,
                3.85168607e-04,
                5.54623615e-04,
                7.54873514e-04,
                9.85910750e-04,
                1.24772661e-03,
            ]
        )
        N_w_true = 1024
        t_w_true = 0.010922666666666667
        t_w_n_true = np.array(
            [
                0.00000000e00,
                1.06666667e-05,
                2.13333333e-05,
                3.20000000e-05,
                4.26666667e-05,
                5.33333333e-05,
                6.40000000e-05,
                7.46666667e-05,
                8.53333333e-05,
                9.60000000e-05,
            ]
        )

        w_tilde_i, N_w, t_w, t_w_n = Calculation.defHanningWindow(c, tau, dr, f_s_dec)

        np.testing.assert_equal(N_w, N_w_true)
        np.testing.assert_almost_equal(t_w, t_w_true, decimal=6)
        np.testing.assert_allclose(w_tilde_i[0:10], w_tilde_i_true, rtol=0, atol=1e-10)
        np.testing.assert_allclose(t_w_n[0:10], t_w_n_true, rtol=0, atol=1e-10)

    def test_chirp(self):
        # Calculation trough implemented methods
        signal = Calculation.chirp(np.arange(0, 5), 1, 5, 2)

        # Ground truth
        gt_signal = np.array([1.0, 0.80901699, -0.80901699, 0.80901699, -0.80901699])

        # Test
        np.testing.assert_allclose(signal, gt_signal, rtol=0, atol=1e-8)

    def test_generateIdealWindowedTransmitSignal(self):
        f_0, f_1, tau, f_s, slope = 90000.0, 170000.0, 0.002047999994829297, 5000, 0.5
        y_tx_n, t = Calculation.generateIdealWindowedTransmitSignal(
            f_0, f_1, tau, f_s, slope
        )

        y_tx_n_true = np.array(
            [
                0.0,
                0.02282123,
                0.29215947,
                0.73558894,
                -0.96984631,
                -0.95121093,
                0.53032985,
                -0.08060687,
                0.11697778,
                0.0,
            ]
        )

        t_true = np.array(
            [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018]
        )

        np.testing.assert_allclose(t, t_true, rtol=0, atol=0.0001)
        np.testing.assert_allclose(y_tx_n, y_tx_n_true, rtol=0, atol=0.0001)

    def test_calcAvgSumQuad(self):
        # Calculation trough implemented methods
        ycq = np.array(
            [
                [1.0 + 10.0j, -1.0 - 10.0j],
                [2.0 + 20.0j, -2.0 - 20.0j],
                [3.0 + 30.0j, -3.0 - 30.0j],
                [4.0 + 40.0j, -4.0 - 40.0j],
            ]
        )
        yc = Calculation.calcAverageSignal(ycq)

        # Ground truth
        gt_yc = np.array([2.5 + 25.0j, -2.5 - 25.0j])

        # Test
        np.testing.assert_allclose(yc, gt_yc, rtol=0, atol=0.0001)

    def test_freqtransf(self):
        testData = scipy.io.loadmat("freqtransf.mat")

        FFTvec = Calculation.freqtransf(
            testData["FFTvecin"].squeeze(),
            testData["fsdec"].squeeze(),
            testData["fvec"].squeeze(),
        )

        np.testing.assert_allclose(
            FFTvec, testData["FFTvec"].squeeze(), rtol=0.01, atol=0.1
        )

    def test_freqtransf2(self):
        FFTvecin = np.array(
            [
                1 + 1j,
                2 + 2j,
                3 + 3j,
                4 + 4j,
                5 + 5j,
                6 + 6j,
                7 + 7j,
                8 + 8j,
                9 + 9j,
                10 + 10j,
            ]
        )
        fsdec = 200
        fvec = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])

        FFTvec = Calculation.freqtransf(FFTvecin, fsdec, fvec)
        FFTvec_true = np.array(
            [
                6.0 + 6.0j,
                6.0 + 6.0j,
                7.0 + 7.0j,
                7.0 + 7.0j,
                8.0 + 8.0j,
                8.0 + 8.0j,
                9.0 + 9.0j,
                9.0 + 9.0j,
                10.0 + 10.0j,
                10.0 + 10.0j,
                1.0 + 1.0j,
            ]
        )
        np.testing.assert_allclose(FFTvec, FFTvec_true, rtol=0.01, atol=0.1)

    def test_calcDecmiatedSamplingRate(self):
        f_s = 1500000.0
        filter_v = [{"D": 12}, {"D": 1}]

        f_s_dec_v = Calculation.calcDecmiatedSamplingRate(filter_v, f_s)

        f_s_dec_v_true = np.array([1500000.0, 125000.0, 125000.0])

        np.testing.assert_allclose(
            np.array(f_s_dec_v), f_s_dec_v_true, rtol=0.01, atol=0.1
        )

    def test_calcFilteredAndDecimatedSignal(self):
        real = [
            2.00998271e-04,
            1.17073429e-03,
            3.62765207e-03,
            7.53827440e-03,
            1.11992722e-02,
            1.14458781e-02,
            5.89462789e-03,
            -3.52923968e-03,
            -8.43538623e-03,
            3.89150949e-03,
            4.35592905e-02,
            1.09778665e-01,
            1.87013358e-01,
            2.49629721e-01,
            2.73741663e-01,
            2.49629721e-01,
            1.87013358e-01,
            1.09778665e-01,
            4.35592905e-02,
            3.89150949e-03,
            -8.43538623e-03,
            -3.52923968e-03,
            5.89462789e-03,
            1.14458781e-02,
            1.11992722e-02,
            7.53827440e-03,
            3.62765207e-03,
            1.17073429e-03,
            2.00998271e-04,
        ]

        imag = [
            -0.00040491,
            -0.00122083,
            -0.00199432,
            -0.00151998,
            0.00117709,
            0.00495307,
            0.00508811,
            -0.00569208,
            -0.03285366,
            -0.07425442,
            -0.1171274,
            -0.1415258,
            -0.12997767,
            -0.07822921,
            0.0,
            0.07822921,
            0.12997767,
            0.1415258,
            0.1171274,
            0.07425442,
            0.03285366,
            0.00569208,
            -0.00508811,
            -0.00495307,
            -0.00117709,
            0.00151998,
            0.00199432,
            0.00122083,
            0.00040491,
        ]

        filter_v = [{}, {}]
        filter_v[0]["h_fl_i"] = np.array([complex(r, i) for r, i in zip(real, imag)])
        filter_v[0]["D"] = 6

        real = [
            9.72948601e-06,
            -7.88519101e-05,
            1.78036556e-04,
            1.90495106e-04,
            -7.13263347e-04,
            6.41078077e-05,
            1.21786073e-03,
            -7.60276278e-04,
            -9.35387972e-04,
            9.45187989e-04,
            1.38946445e-04,
            1.28033542e-04,
            -2.14596599e-04,
            -1.29572302e-03,
            1.23755622e-03,
            9.22028034e-04,
            -1.11272000e-03,
            -1.39718295e-05,
            -9.32173512e-04,
            8.77186714e-04,
            2.16597877e-03,
            -2.37831147e-03,
            -8.49557342e-04,
            9.51249152e-04,
            1.07484899e-04,
            2.54269596e-03,
            -2.45281681e-03,
            -3.00682755e-03,
            3.69360880e-03,
            4.23047168e-04,
            4.42846474e-04,
            -1.06055615e-03,
            -5.01672551e-03,
            5.26773743e-03,
            3.25550348e-03,
            -4.26306250e-03,
            -2.22779136e-07,
            -3.96835897e-03,
            3.99054913e-03,
            7.91625679e-03,
            -9.19111911e-03,
            -2.46031908e-03,
            2.40839017e-03,
            8.76769307e-04,
            1.05500845e-02,
            -1.05979815e-02,
            -1.05169434e-02,
            1.34390881e-02,
            7.96617067e-04,
            5.09587675e-03,
            -6.49765739e-03,
            -2.26986893e-02,
            2.54809279e-02,
            1.27454503e-02,
            -1.67753883e-02,
            -3.40910745e-04,
            -3.39408889e-02,
            3.76148969e-02,
            7.14469627e-02,
            -1.16812408e-01,
            -3.79635058e-02,
            1.59456074e-01,
            -3.79635058e-02,
            -1.16812408e-01,
            7.14469627e-02,
            3.76148969e-02,
            -3.39408889e-02,
            -3.40910745e-04,
            -1.67753883e-02,
            1.27454503e-02,
            2.54809279e-02,
            -2.26986893e-02,
            -6.49765739e-03,
            5.09587675e-03,
            7.96617067e-04,
            1.34390881e-02,
            -1.05169434e-02,
            -1.05979815e-02,
            1.05500845e-02,
            8.76769307e-04,
            2.40839017e-03,
            -2.46031908e-03,
            -9.19111911e-03,
            7.91625679e-03,
            3.99054913e-03,
            -3.96835897e-03,
            -2.22779136e-07,
            -4.26306250e-03,
            3.25550348e-03,
            5.26773743e-03,
            -5.01672551e-03,
            -1.06055615e-03,
            4.42846474e-04,
            4.23047168e-04,
            3.69360880e-03,
            -3.00682755e-03,
            -2.45281681e-03,
            2.54269596e-03,
            1.07484899e-04,
            9.51249152e-04,
            -8.49557342e-04,
            -2.37831147e-03,
            2.16597877e-03,
            8.77186714e-04,
            -9.32173512e-04,
            -1.39718295e-05,
            -1.11272000e-03,
            9.22028034e-04,
            1.23755622e-03,
            -1.29572302e-03,
            -2.14596599e-04,
            1.28033542e-04,
            1.38946445e-04,
            9.45187989e-04,
            -9.35387972e-04,
            -7.60276278e-04,
            1.21786073e-03,
            6.41078077e-05,
            -7.13263347e-04,
            1.90495106e-04,
            1.78036556e-04,
            -7.88519101e-05,
            9.72948601e-06,
        ]

        imag = [
            -2.51365527e-05,
            -5.71274068e-05,
            -1.47468949e-04,
            4.05296159e-04,
            1.36019764e-04,
            -1.02250534e-03,
            3.95779323e-04,
            1.19765266e-03,
            -9.95849841e-04,
            -5.19743946e-04,
            5.42028341e-04,
            1.84407117e-18,
            8.34930921e-04,
            -7.12207286e-04,
            -1.31810072e-03,
            1.45323481e-03,
            3.61472252e-04,
            -2.18534900e-04,
            -1.77779104e-04,
            -1.86459010e-03,
            1.79203786e-03,
            1.72778254e-03,
            -2.14517419e-03,
            -1.20198856e-04,
            -8.52594036e-04,
            1.00681279e-03,
            3.37570603e-03,
            -3.63436085e-03,
            -1.73818530e-03,
            2.21885880e-03,
            2.78755542e-05,
            3.26337083e-03,
            -3.18356929e-03,
            -4.94694384e-03,
            5.92214428e-03,
            1.09451124e-03,
            -5.09491540e-04,
            -1.01884420e-03,
            -7.25918682e-03,
            7.43406964e-03,
            5.83272008e-03,
            -7.57139828e-03,
            -1.51537082e-04,
            -4.59735375e-03,
            4.96459799e-03,
            1.28104892e-02,
            -1.44750243e-02,
            -5.32099977e-03,
            6.30763685e-03,
            6.43787091e-04,
            1.64106470e-02,
            -1.64914019e-02,
            -2.10798401e-02,
            2.70859338e-02,
            3.20003415e-03,
            5.41508012e-03,
            -1.10279908e-02,
            -5.92719465e-02,
            7.60835186e-02,
            6.42180443e-02,
            -1.47857234e-01,
            2.47638569e-20,
            1.47857234e-01,
            -6.42180443e-02,
            -7.60835186e-02,
            5.92719465e-02,
            1.10279908e-02,
            -5.41508012e-03,
            -3.20003415e-03,
            -2.70859338e-02,
            2.10798401e-02,
            1.64914019e-02,
            -1.64106470e-02,
            -6.43787091e-04,
            -6.30763685e-03,
            5.32099977e-03,
            1.44750243e-02,
            -1.28104892e-02,
            -4.96459799e-03,
            4.59735375e-03,
            1.51537082e-04,
            7.57139828e-03,
            -5.83272008e-03,
            -7.43406964e-03,
            7.25918682e-03,
            1.01884420e-03,
            5.09491540e-04,
            -1.09451124e-03,
            -5.92214428e-03,
            4.94694384e-03,
            3.18356929e-03,
            -3.26337083e-03,
            -2.78755542e-05,
            -2.21885880e-03,
            1.73818530e-03,
            3.63436085e-03,
            -3.37570603e-03,
            -1.00681279e-03,
            8.52594036e-04,
            1.20198856e-04,
            2.14517419e-03,
            -1.72778254e-03,
            -1.79203786e-03,
            1.86459010e-03,
            1.77779104e-04,
            2.18534900e-04,
            -3.61472252e-04,
            -1.45323481e-03,
            1.31810072e-03,
            7.12207286e-04,
            -8.34930921e-04,
            -1.79454332e-18,
            -5.42028341e-04,
            5.19743946e-04,
            9.95849841e-04,
            -1.19765266e-03,
            -3.95779323e-04,
            1.02250534e-03,
            -1.36019764e-04,
            -4.05296159e-04,
            1.47468949e-04,
            5.71274068e-05,
            2.51365527e-05,
        ]

        filter_v[1]["h_fl_i"] = np.array([complex(r, i) for r, i in zip(real, imag)])
        filter_v[1]["D"] = 5

        y_tilde_tx_n = [
            0.0,
            0.34811461,
            -0.67880075,
            -0.98265328,
            -0.4539905,
            0.33688985,
            0.872496,
            0.9941745,
            0.809017,
            0.49716327,
            0.19509033,
            -0.02944816,
            -0.15643445,
            -0.18545222,
            -0.11753737,
            0.0490677,
            0.30901702,
            0.62677385,
            0.90814319,
            0.99186669,
            0.70710675,
            0.02944812,
            -0.73432255,
            -0.97882274,
            -0.30901693,
            0.74095118,
            0.87249597,
            -0.28087619,
            -0.98768833,
            0.12728111,
            0.98078526,
            -0.35531101,
            -0.80901692,
            0.83688383,
            0.11753726,
            -0.90398923,
            0.8910066,
            -0.28087626,
            -0.41865958,
            0.86765692,
            -1.0,
            0.91220961,
            -0.73432266,
            0.56370645,
            -0.45399071,
            0.42755532,
            -0.48862147,
            0.62677403,
            -0.80901716,
            0.96507375,
            -0.98078522,
            0.7276229,
            -0.15643413,
            -0.56370655,
            0.9930685,
            -0.67155868,
            -0.30901736,
            0.99417455,
            -0.41865936,
            -0.79135719,
            0.70710647,
            0.68597745,
            -0.6788004,
            -0.83688402,
            0.30901651,
            0.99879543,
            0.48862171,
            -0.48003163,
            -0.98768825,
            -0.79135729,
            -0.19509092,
            0.40972346,
            0.80901662,
            0.97882262,
            0.99306854,
            0.9415443,
            0.89100685,
            0.87725126,
            0.90814349,
            0.96507387,
            1.0,
            0.93474789,
            0.67880013,
            0.1854514,
            -0.45399128,
            -0.94154437,
            -0.87249556,
            -0.10778153,
            0.80901756,
            0.86765652,
            -0.19509131,
            -0.99956634,
            -0.15643343,
            0.98265348,
            0.11753631,
            -0.9987954,
            0.30901808,
            0.77920051,
            -0.90814367,
            0.12728221,
            0.7071059,
            -0.99956635,
            0.15418702,
        ]

        y_tilde_tx_nv = Calculation.calcFilteredAndDecimatedSignal(
            y_tilde_tx_n, filter_v
        )

        y_tilde_tx_nv_true = np.array(
            [
                0.00000000e00 + 0.00000000e00j,
                -1.40819203e-04 - 1.28837898e-04j,
                7.68342130e-04 + 6.48924918e-05j,
                8.99294483e-04 + 5.22712009e-04j,
                2.45176302e-04 - 2.01332967e-03j,
                3.71698393e-04 + 2.30560985e-03j,
                1.38122996e-03 + 5.17686198e-03j,
                3.31852022e-03 + 6.98309057e-03j,
                3.26580827e-03 + 5.93745936e-03j,
                -2.67758799e-03 + 3.32550518e-03j,
                -1.58715658e-02 + 2.97874156e-03j,
                -3.43510413e-02 + 6.86389453e-03j,
                -6.46455122e-02 + 9.49859772e-03j,
                5.69003790e-02 - 1.97483830e-03j,
                3.39395551e-02 + 4.53338753e-02j,
                -1.72214039e-01 - 3.33817874e-01j,
                1.27639310e-01 + 1.43014494e-01j,
                7.24775261e-02 + 3.88119868e-02j,
                3.02598428e-02 + 8.53495878e-03j,
                1.59401811e-03 + 2.16966708e-03j,
                -1.15064365e-02 + 6.59273492e-03j,
                -1.15426222e-02 + 1.09040693e-02j,
                -5.87827350e-03 + 9.85947317e-03j,
                -1.37781892e-03 + 4.75612208e-03j,
                1.18655009e-04 - 3.62965808e-04j,
                -1.02373138e-03 - 1.96525731e-03j,
                -2.66144836e-03 - 2.47502063e-03j,
                5.53722815e-04 + 3.76250996e-04j,
                7.00401275e-06 - 1.34325443e-05j,
            ]
        )

        np.testing.assert_allclose(
            y_tilde_tx_nv_true, y_tilde_tx_nv[-1], rtol=0.01, atol=0.1
        )

    def test_calcNormalizedTransmitSignal(self):
        y_tx_n = np.array([-10, -5, 0, 5, 10, 5, 0, -5, -10, -5])

        normsign = Calculation.calcNormalizedTransmitSignal(y_tx_n)

        normsign_true = np.array([-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5])

        np.testing.assert_allclose(normsign, normsign_true, rtol=0.01, atol=0.1)

    def test_alignAuto_auto_gt_yc(self):
        auto = np.array(
            [
                -0.7 - 0.7j,
                -0.6 - 0.6j,
                -0.7 - 0.7j,
                -0.8 - 0.8j,
                -0.9 - 0.9j,
                1 + 1j,
                0.9 + 0.9j,
                0.8 + 0.8j,
                0.7 + 0.7j,
                0.6 + 0.6j,
                0.5 + 0.5j,
            ]
        )
        yc = np.array(
            [-0.8 - 0.8j, -0.9 - 0.9j, 1 + 1j, 0.9 + 0.9j, 0.8 + 0.8j, 0.7 + 0.7j]
        )

        _auto = Calculation.alignAuto(auto, yc)

        np.testing.assert_equal(
            _auto,
            np.array(
                [
                    -0.8 - 0.8j,
                    -0.9 - 0.9j,
                    1.0 + 1.0j,
                    0.9 + 0.9j,
                    0.8 + 0.8j,
                    0.7 + 0.7j,
                ]
            ),
        )

    def test_PSIfc(self):
        f_0, f_1, n_f_points = 90000.0, 170000.0, 10

        f_m = np.linspace(f_0, f_1, n_f_points)
        psifc = self.calc.calcpsi(0.008511380382023767, 38000.0, f_m)

        ans = [
            0.00151734,
            0.00125682,
            0.00105806,
            0.00090297,
            0.00077964,
            0.00067996,
            0.00059824,
            0.00053041,
            0.0004735,
            0.00042527,
        ]

        np.testing.assert_almost_equal(psifc, ans, 5)

    def test_calcg0_calibrated(self):
        f = np.array(
            [
                90000.0,
                98008.00800801,
                106016.01601602,
                114024.02402402,
                122032.03203203,
                130040.04004004,
                138048.04804805,
                146056.05605606,
                154064.06406406,
                162072.07207207,
            ]
        )

        frqp_freq = np.array([90000.0, 106817.0, 123634.0, 140450.0, 157267.0])

        frqp_gain = np.array([21.29, 26.74, 27.13, 29.23, 30.37])

        g0 = Calculation.calcg0_calibrated(f, frqp_freq, frqp_gain)
        g0_true = np.array(
            [
                134.58603541,
                244.63634673,
                444.67423357,
                490.58433764,
                512.0176183,
                620.86804878,
                781.63530615,
                914.11879741,
                1035.82827691,
                1088.93009333,
            ]
        )
        np.testing.assert_almost_equal(g0, g0_true, 5)

    def test_calcg0_notcalibrated(self):
        f = np.array(
            [
                90000.0,
                98008.00800801,
                106016.01601602,
                114024.02402402,
                122032.03203203,
                130040.04004004,
                138048.04804805,
                146056.05605606,
                154064.06406406,
                162072.07207207,
            ]
        )
        f_n, G_f_n = 120000.0, 27.0
        g0 = Calculation.calcg0_notcalibrated(f, f_n, G_f_n)
        g0_true = np.array(
            [
                281.91781892,
                334.31867168,
                391.18344545,
                452.51214023,
                518.30475602,
                588.56129282,
                663.28175063,
                742.46612945,
                826.11442929,
                914.22665013,
            ]
        )
        np.testing.assert_almost_equal(g0, g0_true, 5)

    def test_calcPower(self):
        # Calculation trough implemented methods
        z_td_e, z_rx_e, N_u = 75, 5400, 4
        y_pc_n = np.array(
            [
                -9.99537661e-05 + 2.97907232e-05j,
                -2.63518109e-05 - 9.63090422e-05j,
                6.27873532e-05 - 3.33523700e-07j,
                -2.04267017e-05 + 3.20007899e-05j,
                -2.94661296e-05 - 1.66837242e-05j,
            ]
        )
        power = self.calc.calcPower(y_pc_n, z_td_e, z_rx_e)

        # Ground truth
        gt_power = np.array(
            [8.3808e-11, 7.6809e-11, 3.0373e-11, 1.1104e-11, 8.8336e-12]
        )

        # Test
        np.testing.assert_allclose(power, gt_power, rtol=0, atol=1e-9)

    def test_calcSv(self):
        p_rx_e_n = np.array(
            [
                3.02100061e01,
                3.10046036e00,
                2.02596796e00,
                2.10450794e-01,
                4.62668222e-01,
                3.87325468e-01,
                3.76095997e-03,
                1.28006039e-01,
                1.18464386e-01,
                3.82320985e-04,
            ]
        )

        r_c_n = np.array(
            [
                1.00000000e-20,
                7.90402492e-03,
                1.58080498e-02,
                2.37120747e-02,
                3.16160997e-02,
                3.95201246e-02,
                4.74241495e-02,
                5.53281744e-02,
                6.32321993e-02,
                7.11362242e-02,
            ]
        )

        lambda_f_c = 0.011856
        p_tx_e = 100
        alpha_f_c = 0.03448625768285417
        c = 1482.0
        tau_eff = 1.5690021364495307e-05
        psi_f_c = 0.007844088160073103
        g_0_f_c = 621.6154220851992

        Sv = Calculation.calcSv(
            p_rx_e_n, r_c_n, lambda_f_c, p_tx_e, alpha_f_c, c, tau_eff, psi_f_c, g_0_f_c
        )

        Sv_true = np.array(
            [
                -360.1633361,
                -12.09307123,
                -7.91986225,
                -14.23231197,
                -8.31180104,
                -7.14499179,
                -25.68859515,
                -9.0297967,
                -8.20583912,
                -32.09384023,
            ]
        )

        np.testing.assert_almost_equal(Sv, Sv_true, 5)

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

    def test_pulseCompression(self):
        y_mf_n = np.array(
            [
                0.00000000e00 + 0.0j,
                8.18771469e-03 - 0.00582447j,
                -3.07992800e-02 - 0.01009615j,
                1.02180327e00 - 0.28663263j,
                -8.36105932e-01 - 0.58982137j,
                -1.00772648e00 + 0.10053897j,
                -7.42972244e-01 - 0.69740986j,
                9.89186451e-01 + 0.01900877j,
                -4.75021920e-02 + 0.04353109j,
                1.74240029e-03 - 0.01309336j,
                7.81659333e-04 - 0.00363211j,
            ]
        )

        y_rx_nu = np.array(
            [
                [
                    -1.17646799e-04 - 3.95109419e-05j,
                    3.74271971e-04 + 3.22387095e-05j,
                    -1.29431006e-04 + 4.91670617e-05j,
                    -3.69502115e-04 - 1.91425075e-04j,
                    3.19951891e-06 + 1.16657147e-05j,
                    1.88612830e-05 + 3.48608628e-05j,
                    -1.04294209e-04 - 2.47430318e-04j,
                    -4.30864093e-05 - 1.34203830e-04j,
                    3.34305478e-05 + 3.43852771e-05j,
                    6.49533977e-05 - 5.79879670e-05j,
                ],
                [
                    -1.98100970e-04 - 4.49943145e-05j,
                    2.85173319e-05 - 3.25633242e-04j,
                    1.53754256e-04 + 2.15744061e-04j,
                    -6.51946175e-05 + 2.57920794e-04j,
                    9.49954338e-05 - 5.73917896e-05j,
                    8.54586251e-05 + 5.51276062e-05j,
                    -1.31343622e-04 + 7.97366447e-05j,
                    -2.17818870e-05 - 1.24406952e-05j,
                    2.62514423e-05 + 3.48618087e-05j,
                    -8.28368793e-05 + 3.86763531e-05j,
                ],
                [
                    -1.06678090e-04 - 6.42403102e-05j,
                    2.69366603e-04 - 2.55059596e-04j,
                    -8.83403663e-06 + 1.95716493e-04j,
                    -2.00008435e-04 + 1.60429816e-04j,
                    1.88088030e-04 + 4.51141786e-05j,
                    1.03579870e-04 + 1.12163390e-04j,
                    -1.69490071e-04 + 3.73643670e-05j,
                    -4.52302629e-05 + 8.25982352e-05j,
                    2.24370451e-05 + 1.21700556e-04j,
                    -3.42654894e-05 + 2.67060682e-06j,
                ],
                [
                    5.72731769e-05 - 1.17646116e-04j,
                    2.59828870e-04 + 2.19559515e-04j,
                    -3.00836167e-04 - 5.60939907e-06j,
                    -2.70318822e-04 - 2.22896357e-04j,
                    3.72458526e-05 + 9.02283864e-05j,
                    2.92327677e-05 - 4.24897917e-05j,
                    -2.63714082e-05 - 3.40891158e-04j,
                    -4.33252244e-05 - 1.59953968e-04j,
                    6.87699358e-05 + 2.41661419e-05j,
                    1.25144259e-04 - 6.99625889e-05j,
                ],
            ]
        )

        yc_q = self.calc.calcPulseCompressedSignals(y_rx_nu, y_mf_n)

        y_rx_nu_true = np.array(
            [
                [
                    -2.59929093e-05 - 7.08515524e-05j,
                    4.01230017e-05 + 6.88343097e-05j,
                    5.44954531e-05 + 5.00020181e-05j,
                    7.75840390e-06 - 2.80227782e-05j,
                    -2.28696158e-05 - 1.90829994e-05j,
                    1.43009407e-06 + 2.53903893e-05j,
                    1.57123588e-05 - 8.17875901e-06j,
                    -2.56938627e-07 + 5.58841100e-07j,
                    1.66534536e-07 - 1.84753072e-08j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    -4.81977983e-05 + 2.11785226e-05j,
                    3.95359007e-05 - 1.34791909e-05j,
                    4.90301426e-06 - 2.27668426e-06j,
                    -2.33582484e-05 - 1.60195143e-05j,
                    5.70369567e-06 - 1.26589658e-05j,
                    1.19765470e-05 - 7.27732686e-06j,
                    -1.85759411e-05 + 2.82262001e-06j,
                    4.16106349e-07 - 3.04353480e-07j,
                    -1.73035461e-07 - 3.17549098e-08j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    -8.89594135e-05 - 3.99596921e-06j,
                    3.95732254e-05 + 1.33507323e-05j,
                    1.97328309e-05 - 2.61265759e-05j,
                    -3.53574024e-05 - 4.93843120e-05j,
                    -2.30959308e-05 - 3.38454569e-06j,
                    2.69480411e-06 + 2.00438241e-05j,
                    -7.38271558e-06 - 1.95376433e-06j,
                    9.63805888e-08 + 1.33854988e-07j,
                    -5.67093483e-08 - 3.40343413e-08j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    -2.10448797e-05 - 4.66392941e-05j,
                    4.14630362e-05 + 1.17446448e-04j,
                    6.59347296e-05 + 6.75266508e-05j,
                    1.71564228e-05 - 2.62658612e-05j,
                    -3.80306135e-05 - 1.67625934e-05j,
                    8.98775918e-07 + 3.41388745e-05j,
                    2.79880739e-05 - 7.12997829e-06j,
                    -5.22006796e-07 + 7.69251172e-07j,
                    2.74274767e-07 + 2.98886629e-08j,
                    0.00000000e00 + 0.00000000e00j,
                ],
            ]
        )

        np.testing.assert_allclose(
            yc_q[0][0:4], y_rx_nu_true[0][0:4], rtol=0.01, atol=0.1
        )

    """
    def test_C1Prx(self):
        np.testing.assert_almost_equal(self.calc.C1Prx, 0.0068531, 5)
    """

    def test_Gf(self):
        Gf = self.calc.calcg0(39500)

        np.testing.assert_almost_equal(Gf, 26.11500, 5)

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
    def test_calcDFTforSv(self);    
        y_pc_s_n=y_pc_s_n[::1000]
        w_tilde_i= w_tilde_i[::130]
        y_mf_auto_n=y_mf_auto_n[::200]
        f_m=f_m[::100]
        r_c_n=r_c_n[::1000]
        N_w = 8
    """

    """
    P_rx_e_t_m_n[0]=P_rx_e_t_m_n[0][0][::100]
    P_rx_e_t_m_n[1]=P_rx_e_t_m_n[1][0][::100]
    P_rx_e_t_m_n[2]=P_rx_e_t_m_n[2][0][::100]
    P_rx_e_t_m_n[3]=P_rx_e_t_m_n[4][0][::100]
    P_rx_e_t_m_n=P_rx_e_t_m_n[0:3]
    alpha_m=alpha_m[::100]
    lambda_m=lambda_m[::100]
    psi_m=psi_m[::100]
    g_0_m=g_0_m[::100]
    svf_range=svf_range[0:3]
    def test_Svf(self):
        ycq = self.calc.calcPulseCompressedSignals(self.calc.y_rx_org)
        yc = self.calc.calcAverageSignal(ycq)
        Svf, svf_range, f = self.calc.calcSvf(yc,10,30)

        np.testing.assert_allclose(Svf[0,:][0:4], [-58.21107559, -58.23425062, -58.25742747, -58.28060613], rtol=0.0001, atol=0.001)
        np.testing.assert_allclose(svf_range,[14.03414407, 18.06188048, 22.0896169 , 26.05441993], rtol=0.0001, atol=0.001)
        np.testing.assert_allclose(f[0:4], [34000., 34011.01101101, 34022.02202202, 34033.03303303], rtol=0.0001,
                                   atol=0.001)
    """

    """
    f=f[::100]
    self.frqp.frequencies=self.frqp.frequencies[::35]
    self.frqp.angle_offset_alongship=self.frqp.angle_offset_alongship[::35]
    self.frqp.angle_offset_athwartship=self.frqp.angle_offset_athwartship[::35]
    self.frqp.beam_width_alongship=self.frqp.beam_width_alongship[::35]
    self.frqp.beam_width_athwartship=self.frqp.beam_width_athwartship[::35]
    self.frqp.gain=self.frqp.gain[::35]
    
    self.isCalibrated = False
    
    """


if __name__ == "__main__":
    unittest.main()
