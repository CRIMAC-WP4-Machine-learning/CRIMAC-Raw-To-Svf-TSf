import matplotlib.pyplot as plt
import numpy as np
import argparse

from Core.EK80Calculation import EK80Calculation

# python example_Svf.py --file ..\Data\pyEcholabEK80data.json --r0 10 --r1 30
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do signal processing on ek80 data')
    parser.add_argument('--file', metavar='File', type=str,
                        help='File where data is stored')
    parser.add_argument('--r0', type=float, help='Start range in meters')
    parser.add_argument('--r1', type=float, help='End range in meters')

    args = parser.parse_args()

    #ekdata = EK80DataContainer(args.file)
    ekcalc = EK80Calculation(args.file)

    y_pc_u = ekcalc.calcPulseCompressedQuadrants(ekcalc.y_rx_nu)
    y_pc = ekcalc.calcAvgSumQuad(y_pc_u)
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



