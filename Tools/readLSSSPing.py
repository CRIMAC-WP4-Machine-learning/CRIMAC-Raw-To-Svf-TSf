# -*- coding: utf-8 -*-
"""
Read data with LSSS - select a depth/sample range within a single ping
Sv, pulse compressed/complex, TSc
Write to json
For comparison with the EK80Processing scripts
LSSS needs to be open and have a working data directory with the correct data
file(s)
Ping number is "relative ping number", not the ping number in the raw file
Using Python numbering (0=1, 1=2, etc.).
GP 01.06.2021
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import lsss
import json
import math
import requests
import numpy as np

baseUrl = "http://127.0.0.1:8000"


# dongle=lsss.get('/korona/cli/print-datagrams')

# =============================================================================
# ##### Read LSSS data
# # Note: The relevant windows must be initialized in LSSS
# # Numerical view must show data
# # All BB calibration value boxes must be ticked
#
# # Which channel, ping, depth range to read. Depth range affects the SV and TS of f
# =============================================================================
channel = 1  # 0 - 18kHz, 1 - 38 kHz,..., assuming all frequencies are pressent
minDepth = 21.5  # Approximate depth - will be adjusted by the sample resolution
maxDepth = 24.5  # Approximate depth - will be adjusted by the sample resolution
pingNoRel = 11  # Note. The raw file ping may not begin at 1. So PingNo and PingNoRel may or may not be equal

default = 0

# =============================================================================
# # Zoom out the echogram and find what the number of the first ping is
# # add the selected ping number
# # zoom out max
# =============================================================================
max_zoom = lsss.get("/lsss/module/PelagicEchogramModule/zoom/max")
pingNo = max_zoom[0]["pingNumber"] + pingNoRel - 1
minDepthMax = max_zoom[0]["z"]
maxDepthMax = max_zoom[1]["z"]

zoomRegion = [
    {"pingNumber": max_zoom[0]["pingNumber"]},
    {"pingNumber": max_zoom[1]["pingNumber"]},
]
requests.post(baseUrl + "/lsss/module/PelagicEchogramModule/zoom", json=zoomRegion)

# =============================================================================
# # Get sample resolution, vertical
# =============================================================================
tmp = lsss.get(
    "/lsss/data/ping",
    params={
        "pingNumber": pingNo,
        "minDepth": minDepth,
        "maxDepth": maxDepth,
        "sv": True,
    },
)
sampleDistance = tmp["channels"][channel]["sampleDistance"]

# =============================================================================
# # Get the actual depth and sample number to use when subsetting the ping
# # This information will also be stored in the json file
# =============================================================================

minSample = math.floor(minDepth / sampleDistance)
maxSample = math.ceil(maxDepth / sampleDistance)

minDepth = minSample * sampleDistance
maxDepth = maxSample * sampleDistance

# =============================================================================
# Usefull routine
# =============================================================================


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step


# =============================================================================
# ##### Calibration values
# =============================================================================

cal = lsss.get("/lsss/module/BroadbandCalibrationPlotModule/data")

# =============================================================================
# ##### Time of the first ping?
# =============================================================================
start_time = max_zoom[0]["time"]

# =============================================================================
# Make schoolbox for single ping with dept ranges given
# For Svf and TSf extraction
# =============================================================================

lsss.post("/lsss/package/lsss/action/resetInterpretation/run")

example = {"pingNumber": pingNo, "depthRanges": [{"min": minDepth, "max": maxDepth}]}
lsss.post("/lsss/module/PelagicEchogramModule/school-mask", json=example)

dict = {"ids": []}

regionId = lsss.get("/lsss/regions/region")
n = len(regionId)
for i in my_range(0, n - 1, 1):
    infoRegion = lsss.get("/lsss/regions/region/" + str(regionId[i]["id"]))

    # Is the region a school?
    if infoRegion["type"] == "school":
        ids = regionId[i]["id"]
        dict["ids"].append(ids)

# Selects all regions
op = {"operation": "SET", "schools": "true"}
lsss.post("/lsss/regions/selection", json=op)

# =============================================================================
# Extract Svf TSf
# =============================================================================
requests.get(baseUrl + "/lsss/data/wait")  # Just make sure LSSS is ready
lsss.post(
    "/lsss/module/BroadbandSvModule/enabled", json={"value": True}
)  # Just make sure the BB Sv(f) window is active
params = lsss.get(
    "/lsss/module/BroadbandSvModule/config/parameter"
)  # What can you get out, and what are the settings
BroadbandSv = lsss.get("/lsss/module/BroadbandSvModule/data")
lsss.post(
    "/lsss/module/BroadbandTsModule/enabled", json={"value": True}
)  # Just make sure the BB Sv(f) window is active
BroadbandTS = lsss.get("/lsss/module/BroadbandTsModule/data")

# =============================================================================
# ##### Sv values - read from LSSS value in ping between minDepth, maxDepth
# =============================================================================
Sv = lsss.get(
    "/lsss/data/ping",
    params={
        "pingNumber": pingNo,
        "minDepth": minDepth,
        "maxDepth": maxDepth,
        "sv": True,
    },
)
##### Pulse compressed values - read from LSSS
PC = lsss.get(
    "/lsss/data/ping",
    params={
        "pingNumber": pingNo,
        "minDepth": minDepth,
        "maxDepth": maxDepth,
        "pulseCompressed": True,
    },
)

##### TSc values - read from LSSS
TSc = lsss.get(
    "/lsss/data/ping",
    params={
        "pingNumber": pingNo,
        "minDepth": minDepth,
        "maxDepth": maxDepth,
        "tsc": True,
    },
)
#### Complex values - pre PC
PPC = lsss.get(
    "/lsss/data/ping",
    params={
        "pingNumber": pingNo,
        "minDepth": minDepth,
        "maxDepth": maxDepth,
        "complex": True,
    },
)

##### Print some info to screen
print(PC["channels"][channel])  # print complex data

# =============================================================================
# ##### Plot LSSS ping Sv vs sample no.
# Plot Svf
# =============================================================================
fig = figure()
svplt = plt.plot(range(minSample, maxSample, 1), Sv["channels"][channel]["sv"])
plt.title("LSSS - Sv")
plt.grid()
plt.ylim(-150, 10)
plt.xlabel("Sample")
plt.ylabel("Sv")
show()

tmpf = np.array(BroadbandSv["datasets"][0]["frequency"])
tmps = np.array(BroadbandSv["datasets"][0]["sv"])
fig = figure()
svplt = plt.plot(tmpf, tmps)
plt.title("LSSS - Sv")
plt.grid()
# plt.ylim(-150, 10)
plt.xlabel("Sample")
plt.ylabel("Frequency")
show()


# =============================================================================
# Cal values needed for json
# =============================================================================
cal_freq = np.array(cal["datasets"][1]["frequency"]) * 1000
cal_freq = np.ndarray.tolist(cal_freq)
# =============================================================================

###############################################################################
# Dict for jsonwrite
###############################################################################

data = {
    "COMMENTS": {
        "Free text1": "Experiment file from CRIMAC 2020 cruise, FM 38 kHz (channel 1) with two spheres ",
        "Free text2": "Calibration sphere WC22 mm at 21.7 meter depth +/-",
        "Free text3": "Calibration sphere WC57.2 mm at 27.8 meter depth +/-",
        "Free text4": "Seaflor at 41 meter depth +/-",
        "Free text5": "Start range/depth 8m +/-",
        "Free text": "Single ping (see PingNo / PingTime for identifier)",
        "Raw file": fileName,
        "json file": "LSSSdataPulseCompressedv2.json",
    },
    "XML0": {
        "fs": 1500000,
        "FrequencyStart": dir["startFrequency"],
        "FrequencyEnd": dir["endFrequency"],
        "Frequency": dir["transducer"]["frequency"],
        "PulseDuration": dir["channelData"]["pulseDuration"],
        "SampleInterval": dir["channelData"]["sampleInterval"],
        "Slope": dir["filterSlope"],
        "TransmitPower": dir["channelData"]["transmitPower"],
        "Temperature": 0,
        "Salinity": 0,
        "Alpha": dir["channelData"]["absorption"],
        "SoundSpeed": dir["channelData"]["soundVelocity"],
        "Gain": dir["transducer"]["gain"],
        "EquivalentBeamAngle": dir["transducer"]["equivalentBeamAngle"],
        "SaCorrection": dir["transducer"]["saCorrection"],
        "PingNo": pingNoRel,
        "PingTime": dir["channelData"]["time"],
        "DropKeelOffset": dir["channelData"]["heaveCorrectedTransducerDepth"],
    },
    "FIL1": "NaN",
    "RAW3": {
        "offset": PC["channels"][1]["offset"],
        "sampleCount": len(PC["channels"][1]["pulseCompressed"]["re"]),
        "quadrant_signals": "NaN",
        "yc": {
            "real": PC["channels"][1]["pulseCompressed"]["re"],
            "imag": PC["channels"][1]["pulseCompressed"]["im"],
        },
        "Svf": {
            "Svf": np.ndarray.tolist(np.array(BroadbandSv["datasets"][0]["sv"])),
            "f": np.ndarray.tolist(np.array(BroadbandSv["datasets"][0]["frequency"])),
        },
        "TSf": {
            "TSf": np.ndarray.tolist(np.array(BroadbandTS["datasets"][0]["tsc"])),
            "f": np.ndarray.tolist(np.array(BroadbandTS["datasets"][0]["frequency"])),
        },
    },
    "CAL": {
        "frequencies": cal_freq,
        "gain": cal["datasets"][1]["gain"],
        "angle_offset_athwartship": cal["datasets"][10]["angleOffsetAthwartship"],
        "angle_offset_alongship": cal["datasets"][8]["angleOffsetAlongship"],
        "beam_width_athwartship": cal["datasets"][6]["beamWidthAthwartship"],
        "beam_width_alongship": cal["datasets"][4]["beamWidthAlongship"],
        "drop_keel_offset": dir["channelData"]["heaveCorrectedTransducerDepth"],
    },
}

###############################################################################
# Write to json
###############################################################################
with open("../Data/LSSSEK80data.json", "w") as outfile:
    json.dump(data, outfile, indent=2)
