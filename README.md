# CRIMAC-Raw-To-Sv(f)-TS(f)
Python code illustrating the current basic steps for handling broadband echosounder data from raw data (exemplified using the EK80-family data) to frequency dependent volume backscattering strength (Sv(f)) and target strength (TS(f)).

Example broadband data consisting of complex raw data from one ping collected with a broadband echosounder (EK80) operating with 120 kHz centre frequency  is included and stored in json format following the structure of a raw file. The example data are located in \Data and one for calculation of TS(f) - CRIMAC_SphereBeam.json and one for calculation of Sv(f) - CRIMAC_Svf.json.

The example raw data can be processed to frequency dependent Sv and TS using main.py, which also calls plots.py to reproduce the figures in Andersen et al.

The majority of the calculations are performed in Calculation.py found under \Core 

Requirements to run the code are given in requirements_win.txt

## main.py

main.py initiates the calculations, and generate and plots the figures from the accompagning paper by Andersen et al., using two broadband pings stored in json format, one for TS(f) calculation and one for Sv(f).
TSf.py initiates calculations for TS(f), and generate and plots figures
Svf.py initiates calculations for Sv(f), and generate and plots figures 
Tools\SvfEchogram.py - reads raw data (EK80) and reproduces the Sv echogram in the accompagning paper by Andersen et al. (if given the correct raw file)
Tools\TSfEchogram.py - reads raw data (EK80) and reproduces the Sp echogram in the accompagning paper by Andersen et al. (if given the correct raw file)



