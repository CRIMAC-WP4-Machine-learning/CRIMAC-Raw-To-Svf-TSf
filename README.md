# CRIMAC-Raw-To-Sv(f)-TS(f)
Python code illustrating the current basic steps for handling broadband echosounder data from raw data (exemplified using the EK80-family data) to frequency dependent volume backscattering strength (Sv(f)) and target strength (TS(f)).

Example broadband data consisting of complex raw data from one ping callected with the EK80 is included and stored in json format following the structure of a raw file. The example data is located in \Examples

The example raw data can be processed to frequency dependent Sv and TS (to be implemented) using example scripts located in \Examples.

The majority of the calculations are performed in EK80Calculation.py found under \Core 

Requirements to run the code are given in requirements_win.txt

## example_sendPulse.py

example_sendPulse.py generate and plots an ideal enveloped transmit pulse and is run with:

python example_sendPulse.py --file ..\Data\pyEcholabEK80data.json

This example uses the broadband data file and EK80Calculation.py to calculate the ideal transmit pulse used for pulse compression.

## example_Svf.py

example_Svf.py processes the raw data to Sv(f) and can be run using

python example_Svf.py --file ..\Data\pyEcholabEK80data.json --r0 10 --r1 30

where the values given after --r0 and --r1 is the range in meters from the transducer. In the example above Sv(f) is calculated over depth range 10 to 30m.
Note that the actual values used for range depends on the length of the Hann window used, the code uses number of samples corresponding to twice the pulse duration (see EK80Calculation.py).

The script also produces three plots
1. An echogram showing range, acoustic frequency and Sv, overlayed Sv as a function of range
2. Sv(f) in depth bins between r0 and r1
3. Sv(f) over the entire depth layer

## example_TSf.py

To be implemented



