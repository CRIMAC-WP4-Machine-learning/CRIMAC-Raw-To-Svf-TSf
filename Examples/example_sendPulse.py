import matplotlib.pyplot as plt
import argparse

from Core.EK80DataContainer import EK80DataContainer
from Core.EK80Calculation import EK80Calculation

# This example script generates and plots an ideal enveloped transmit pulse
# A replica transmit pulse is used as the matched filter in the puls compression processing
# python example_sendPulse.py --file ..\Data\pyEcholabEK80data.json

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do signal processing on ek80 data')
    parser.add_argument('--file', metavar='File', type=str,
                        help='File where data is stored')

    args = parser.parse_args()

    ekcalc = EK80Calculation(args.file)


    plt.figure()
    plt.plot(ekcalc.ytx0_t, ekcalc.ytx0)
    plt.title('Ideal enveloped transmit pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'.format(ekcalc.f0/1000, ekcalc.f1/1000, ekcalc.slope))
    plt.xlabel('sec')
    plt.ylabel('amplitude')
    plt.show()
