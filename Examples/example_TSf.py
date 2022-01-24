import numpy as np
import argparse

from Core.EK80DataContainer import EK80DataContainer
from Core.EK80Calculation import EK80Calculation

# python example1.py --file ..\Data\pyEcholabEK80data.json --r0 400 --r1 420
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do signal processing on ek80 data')
    parser.add_argument('--file',required=True, metavar='File', type=str,
                        help='File where data is stored')
    parser.add_argument('--r0',required=True, type=float, help='Start range in meters')
    parser.add_argument('--r1',required=True, type=float, help='End range in meters')

    args = parser.parse_args()

    ekdata = EK80DataContainer(args.file)
    ekcalc = EK80Calculation(ekdata)

    ycq = ekcalc.calcPulseCompressedQuadrants(ekdata.y_rx_org)
    yc = ekcalc.calcAvgSumQuad(ycq)
    power = ekcalc.calcPower(yc)

    r, dr = ekcalc.calcRange()
    idx = np.arange(np.floor(args.r0/dr), np.ceil(args.r1/dr),dtype=int)

    peakIdx = idx[round(len(idx)/2)]-idx[0]

    TSf = ekcalc.calcTSf(yc[idx] , peakIdx)
    """
    plt.figure()
    ax1 = plt.gca()
    ax1.imshow(Svf.T, extent=[r[0], r[-1], f[0] / 1000, f[-1] / 1000],origin='lower')
    ax1.axis('auto')
    ax1.set_ylabel('kHz')
    plt.xlabel('Range(m)')
    ax2 = ax1.twinx()
    ax2.plot(r, Sv,'w', alpha=0.2)
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


    plt.show()
    """


