import matplotlib.pyplot as plt
import numpy as np
import argparse

from Core.EK80DataContainer import EK80DataContainer
from Core.EK80Calculation import EK80Calculation

# python example_TSf.py --file ..\Data\pyEcholabEK80data.json --r0 10 --r1 30 --before 0.5 --after 1
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do signal processing on ek80 data')
    parser.add_argument('--file', required=True, metavar='File', type=str,
                        help='File where data is stored')
    parser.add_argument('--r0', required=True, type=float, help='Start range in meters')
    parser.add_argument('--r1', required=True, type=float, help='End range in meters')
    parser.add_argument('--before', required=True, type=float, help='Range before target in meters')
    parser.add_argument('--after', required=True, type=float, help='Range after target in meters')

    args = parser.parse_args()

    #ekdata = EK80DataContainer(args.file)
    ekcalc = EK80Calculation(args.file)

    y_pc_nu = ekcalc.calcPulseCompressedQuadrants(ekcalc.y_rx_nu)
    y_pc_n = ekcalc.calcAvgSumQuad(y_pc_nu)
    p_rx_e_n = ekcalc.calcPower(y_pc_n)
    theta_n, phi_n = ekcalc.calcElectricalAngles(y_pc_nu)

    Sp_n, r_n = ekcalc.calcSp(p_rx_e_n, args.r0, args.r1)

    r, theta, phi, y_pc_t_n = ekcalc.singleTarget(y_pc_n, p_rx_e_n, theta_n, phi_n,
                                                  args.r0, args.r1, args.before, args.after)

    TS_m, f_m,  = ekcalc.calcTSf(r, theta, phi, y_pc_t_n)

    plt.figure()
    ax1 = plt.gca()
    plt.xlabel('Range [m]')
    ax1.set_ylabel('Sp [dB]')
    ax1.plot(r_n, Sp_n, 'k', alpha=0.3)
    ax1.axis('auto')
    plt.title('Sp_n')

    plt.figure()
    plt.plot(f_m / 1000, TS_m)
    plt.xlabel('kHz')
    plt.ylabel('dB')
    plt.title('TSf')

    plt.show()




