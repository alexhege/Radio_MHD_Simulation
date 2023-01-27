# Radio_MHD_Simulation

Contains code to do analyses from Hegedus et al.

Tracking the Source of Solar Type II Bursts through Comparisons of Simulations and Radio Data

https://iopscience.iop.org/article/10.3847/1538-4357/ac2361

First, dataPrep.py should be run once.

This file does the initial reading of the .dat files, parses them, calculates some secondary variables, and puts the coordinates in a common frame.
It then exports the various data fields to a txt file that is then read in by later scripts for each dat file.  
The script also makes initial plots of 2D and 3D distributions of data.  The script has code to download and unzip the old files with paths that are no longer valid, 
so they will have to be updated with new paths.  
You should only have to run this script once to create the txt files containing all the variables that get read in by other scripts. 

The other 3 scripts here then create various other plots using this data.
Wind data is also used to compare the simulated data to real radio data.  

The code currently expects IDL sav files for Rad 1 and Rad 2 for 20050513, 20050514, 20050515.  It also defines a specific stencil that fits the particular radio burst
in this time period.

This data was downloaded from
https://solar-radio.gsfc.nasa.gov/wind/data_products.html

Each of the following scripts works by defining a set of "cuts"
varInd, varLow, varHigh, label, datPre, mode, bitInd = cut

varInd chooses what variable is being used to define the cut, an integer 0 indexed to choose from the following ordered list of variables
# (x, y, z, ux, uy, uz, Bx, By, Bz, f, n1, n2, n3, fv1, fv2, fv3, ft, thetabn, Nx, Ny, Nz, NDiff, p (pressure), npsingle (electron density), rho (density), ent, entrat, fluxf, fluxh)

varLow and varHigh set the range of the variable to be selected

label is arbitrary text that will go in the title of the generated plots

datPre is the prefix of the data file to be used, eg 'IsoEntropy=4_' or 'isoNe=3.5_'

mode selects how the data will be reduced, custom data reductions are used for Density derived shock 'Lobe', and Entropy derived shock 'Front', 
'FrontBit' selects a single 3x3 degree region of 'Front', etc.

bitInd is used for 'FrontBit' type modes to select which 3x3 degree patch is selected

Different scripts to make different figures:

1. correlateSpectra.py

This script is a main workhorse, it creates a set of synthetic spectra over time for a set of data cuts, and scores its similarity with the stencil.

2. correlateSpectraHeatmap.py

This script is a variant of correlateSpectra.py
It has a custom set of cuts that go over a 40x40 grid of bits to make a similarity score map over the entire shock front.  
Each "bit" is a 3x3 degree patch of solar longitude and latitude.  A similarity score is calculated for the subregion of the data, then plotted in a heatmap.

3. varHeatmap.py

This is another work horse script, instead of calculating a scored synthetic spectra, it creates
heatmaps of chosen variables over the surface of the event for a given time.  Here, the bitInd is used as the minute of data used from the simulation/ which txt file.
Use mode = 'FrontBitVAR' only
