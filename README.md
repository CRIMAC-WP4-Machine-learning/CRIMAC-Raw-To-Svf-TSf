# CRIMAC-Raw-To-Sv(f)-TS(f)
Python code illustrating the current basic steps for handling broadband echosounder data from raw data (exemplified using the EK80-family data) to frequency dependent volume backscattering strength (Sv(f)) and target strength (TS(f)).

Example broadband data consisting of complex raw data from one ping callected with a broadband echosounder (EK80) is included and stored in json format following the structure of a raw file. The example data are located in \Data.

The example raw data can be processed to frequency dependent Sv and TS using main.py.

The majority of the calculations are performed in Calculation.py found under \Core 

Requirements to run the code are given in requirements_win.txt

## main.py

main.py initiates the calculations, and generate and plots the figures from the accompagning paper (xxx,xxx), using two broadband pings stored in json format, one for TS(f) calculation and one for Sv(f).




