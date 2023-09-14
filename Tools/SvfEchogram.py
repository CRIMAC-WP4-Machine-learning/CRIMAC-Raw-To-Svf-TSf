# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:11:12 2020
Script which reads raw EK80 data and makes the Sv echogram for the JEE paper
The script requires pyEcholab (RHT-EK80 branch)
The raw files used in the paper are avilable from
https://zenodo.org/record/8318274
@author: Geir Pedersen, IMR
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap
import sys
sys.path.append(__file__ + '/../../../../../include')
import numpy as np
import sys
from echolab2.instruments import EK80
from echolab2.plotting.matplotlib import echogram
import glob


from datetime import datetime
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go
#import plotly.express as px
import plotly.io as pio
#from time import time
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from plotly.subplots import make_subplots
pio.renderers.default = 'png'

# Raw file to read, change this to reflect raw file and where it is stored
file = 'D:/CRIMAC/CRIMAC_WP1/Svf_new/IMR-D20210507-T074652-Svf.raw'

ek80 = EK80.EK80()

ek80.read_raw(file)
ek80.__dict__

# # read raw data from second channel (vertical echosounder)
raw_list = ek80.raw_data[ek80.channel_ids[4]]
raw_data = raw_list[0]

# del ek80

# get calibration values
calibration = raw_data.get_calibration()

#  convert to Sv
cal_obj = raw_data.get_calibration()
Sv = raw_data.get_Sv(calibation=cal_obj)

# plot a ping
fig1 = figure()
sv=plt.plot(Sv[0,:])
show()

# go from pyEcholab to array
dfSv=Sv.data
dfRange=Sv.get_v_axis()
dfPingTime=Sv.ping_time

# disregard what's after the surface
#dfSv=dfSv[:,0:1400]
dfRange=dfRange[0]
#dfRange=dfRange[0:1400,]
temp=dfSv
temp=np.rot90(dfSv,45)
# temp=np.flipud(temp)
# dfsv=np.power(10,temp/10)

# Plot echogram
fig2 = figure()
pingNo=np.arange(1,len(dfPingTime)+1,1)
indices = np.where(np.logical_and(pingNo >= 400, pingNo <= 650))
dfRange=dfRange[::-1]
plt.pcolormesh(pingNo[indices],dfRange,temp[:,indices[0]],cmap=plt.get_cmap('viridis'), shading='auto', vmin=-82, vmax=-30)
plt.ylim(4.046860756818205,70.94652764296916)
plt.gca().invert_yaxis()
cb=plt.colorbar()
cb.set_label('$S_v$ (dB re 1 m$^{-1}$)')
plt.axvline(x=514,color='red')
# plt.title('Echogram [Sv]')
plt.xlabel('Ping number')
plt.ylabel('Range (m)')
plt.savefig('C:/Users/a32685/Documents/PythonScripts/CRIMAC-WP4-Machine-learning/CRIMAC-Raw-To-Svf-TSf/Paper//Fig_Sv_echogram.png', dpi=300)
plt.show()
