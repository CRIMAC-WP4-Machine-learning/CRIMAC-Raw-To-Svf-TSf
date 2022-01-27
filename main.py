import matplotlib.pyplot as plt
import numpy as np
from Core.EK80Calculation import EK80Calculation
# from Core.EK80DataContainer import EK80DataContainer

# Test data

# The data file ./data/pyEcholabEK80data.json contain data from one
# ping from the EK80 echosounder, including the 

ekcalc = EK80Calculation('./data/pyEcholabEK80data.json')

# Chapter: Signal generation

f0 = ekcalc.f0  # frequency (Hz)
f1 = ekcalc.f1  # frequency (Hz)
tau = ekcalc.tau  # time (s)
fs = ekcalc.f_s  # frequency (Hz)
slope = ekcalc.slope
ytx, t = EK80Calculation.generateIdealWindowedSendPulse(f0, f1, tau, fs, slope)
ytx_05slope, t = EK80Calculation.generateIdealWindowedSendPulse(f0, f1, tau,
                                                                fs, .5)
plt.figure()
plt.plot(t*1000, ytx, t*1000, ytx_05slope)
plt.title(
    'Ideal enveloped send pulse.{:.0f}kHz - {:.0f}kHz, slope {:.3f}'
    .format(f0/1000, f1/1000, slope))
plt.xlabel('time (ms)')
plt.ylabel('amplitude')
plt.savefig('Fig_ytx.png')

# Chapter: Signal reception

# The raw signal 'y_rx_org' is stored as complex numbers in a numpy nd array
# where the first dimension corresponds to one of the four the transducer
# quadrants adn the second is the sample in time. The complex numbers include
# the phase of the signal.
# (or is this the decmiated signal? Unclear.)

print('The four quadrants and the length of the sampled data: ' +
      str(np.shape(ekcalc.y_rx_org)))

# Chapter: Pulse compression

# The source signal filtered through the final decimation is stored in
# ekcalc.y_rx_org
plt.figure()
plt.plot(np.abs(ekcalc.y_mf_n))
plt.title('The absolute value of the filtered and decmiated output signal')
plt.xlabel('samples ()')
plt.ylabel('amplitude')
plt.savefig('Fig_y_mf_n.png')

# The autocorrelation function of the matched filter signal
# ekcalc.y_mf_auto

plt.figure()
plt.plot(np.abs(ekcalc.y_mf_auto))
plt.title('The autocorrelation function of the filtered and decmiated output signal')
plt.xlabel('samples')
plt.ylabel('ACF')
plt.savefig('Fig_ACF.png')


# Calculating the pulse compressed quadrant signals separately on each channel
y_pc_nu = ekcalc.calcPulseCompressedQuadrants(ekcalc.y_rx_org)

# Calculating the average signal over the channels
y_pc_n = ekcalc.calcAvgSumQuad(y_pc_nu)

# Calculating the average signal over paired channels

# This is done inside the calcElectricalAngles(self, y_pc) function and
# the results are not directly accessible.

#
# Chapter: Power angles and samples
#

p_rx_e = ekcalc.calcPower(y_pc_n)

y_theta_n, y_phi_n = ekcalc.calcElectricalAngles(y_pc_nu)


