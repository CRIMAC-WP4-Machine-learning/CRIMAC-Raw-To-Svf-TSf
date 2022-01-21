# CRIMAC-Raw-To-Sv(f)-TS(f)
Python code illustrating the current basic steps for handling broadband echosounder data from raw data (exemplified using the EK80-family data) to Sv(f) and TS(f).
Example data (one ping of broadband data) is included, and can be processed using:

A script to read a EK80 raw data file for processing of one ping is also provided.
readpyecholabPing.py provides a mean to read an EK80 raw data file (using low-level code from pyEcholab), select a single ping from a spesific frequency and store the data in a jason data-file.

Usage: python readpyecholabPing --channel 1 --pingNo 10 --inputFile C:/Users/a32685/Documents/Projects/2020_CRIMAC/CRIMACHackEx/cal-babak-D20201120-T080856.raw --outputFile 'data.json'

Input arguments are 1: Channel(/frequency) to choose 2: Ping number to choose 3: Input path and filename 4: Output filename

For help use: python readpyecholabPing.py -h

This script uses pyEcholab which needs to be installed in order to use EK80 data in the examples

git clone https://github.com/CI-CMG/pyEcholab.git
cd pyEcholab
git checkout RHT-EK80
python setup.py install
To only run EK80 reader follwoing packges must also be installed: a. pip install lxml b. pip install future c. pip install pytz
