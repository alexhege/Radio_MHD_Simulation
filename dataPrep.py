
'''
This file does the initial reading of the .dat files, parses them, and exports the various data
fields to a txt file that is then read in by later scripts
also makes initial plots
has code to download the old files with paths that are no longer valid, so will have to be updated with new paths
you should only have to run this script once 
'''


import numpy as np
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import matplotlib.image as plimg
import scipy.misc

from scipy.special import gamma as gammaFunc
from scipy.special import erf as errFunc
import json
from scipy import interpolate
#apply rotation matrix to positions if needed

from numpy import cross, eye, dot
from scipy.linalg import expm, norm
import scipy.linalg as linalg

import bisect
#import scipy.imageio.imwrite

import subprocess



#import the required python modules
import time
import os,sys
import shutil
from collections import defaultdict
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import numpy as np
import random
import math
import signal
import time
import argparse

from pylab import *

import datetime
import matplotlib.image as plimg
import scipy.ndimage.interpolation as spndint

from matplotlib.patches import Ellipse


#need python 3.6 up
import astropy
import sunpy

import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from astropy.coordinates import CartesianRepresentation

from astropy.time import Time

#SC out to 20 solar radii is HGS coordinates, rotates with sun, IH is HGI, inertial coordinate system
# use astropy to convert from whatever to GSE where X is Earth Sun line


####################################################

def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

import itertools
#list S, number in subset m
def findsubsets(S,m):
    return set(itertools.combinations(S, m))
#then do s.pop()

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


#now find vector derivatives of each component from numerically comparing nearby points
def getVectorDel(N, pos, VectField):
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='kd_tree').fit(pos)
    distances, indices = nbrs.kneighbors(pos)
    #>>> indices = Nx8, distances = Nx8

	#allocate space for the delBperp eg, return later
    delField = np.zeros([N, 3])

    for i in range(N):
        sub = indices[i]
        inds = sub[1:]
        coordSamps = np.zeros(3)

        for j in range(len(inds)):
	                #choose nearby point, calculate approx derivatives for point in question
            subind = inds[j]
            diff = pos[subind]-pos[i]

            for k in range(3):
                if diff[k] == 0.:
                    continue


                diffC = (VectField[subind][k] - VectField[i][k])/(diff[k])
                delField[i][k] += diffC
                coordSamps[k] += 1


	        #average norm
        delField[i] /= coordSamps

    return delField



def getThetaRN(pos, N, B):

    #now calculate shock norm difference to radial direction for every point

    #algorithm kd_tree for fast, or brute for slow
    #http://scikit-learn.org/stable/modules/neighbors.html
    nbrs = NearestNeighbors(n_neighbors=15, algorithm='kd_tree').fit(pos)
    distances, indices = nbrs.kneighbors(pos)

    #>>> indices = Nx8, distances = Nx8

    #calculate shock surface norm, and difference to radial direction
    norms = np.zeros([N, 3])
    normRadDegDiff = np.zeros(N)
    thetaBN = np.zeros(N)



    print('Calculating approximate vector norms & diffs from radial')

    for i in range(N):

        #if i%200==0:
        #    print('tic \n')
        sub = indices[i]

        inds = sub[1:]

        radDir = pos[i]/scipy.linalg.norm(pos[i])

        totalCross = 0

        chooseTwo = findsubsets(inds, 2)
        manyChoose = len(chooseTwo)

        #compute normal derivative from each of 3 triangles made with center point
        for j in range(5): #manyChoose):
        # while norm(norms[i]) == 0.:
            #choose 2 from set of 3, and add in point of interest, define triangle
            sub1, sub2 = chooseTwo.pop()

            vect1 = pos[sub1] - pos[i]
            vect2 = pos[sub2] - pos[i]

            if scipy.linalg.norm(vect1) == 0. or scipy.linalg.norm(vect2) == 0.:
                continue

            vect1 /= scipy.linalg.norm(vect1)
            vect2 /= scipy.linalg.norm(vect2)

            cross = np.cross(vect1, vect2)
            if scipy.linalg.norm(cross) == 0.:
                # print(i, j)
                # print(cross)
                # print(vect1, vect2)
                # print(sub)
                # print(distances[i])
                # print(pos[inds])
                continue


            cross = cross/scipy.linalg.norm(cross)

            #check sign so not adding norm that's inward radially
            normRadDegDiffCheck = np.arccos(np.dot(cross, radDir))/np.pi*180.
            if normRadDegDiffCheck > 90:
            	cross *= -1.

            norms[i] += cross
            totalCross += 1


        #average norm
        norms[i] /= float(totalCross)
        norms[i] /= scipy.linalg.norm(norms[i])

        #radDir = pos[i]/norm(pos[i])

        normRadDegDiff[i] = np.arccos(np.dot(norms[i], radDir))/np.pi*180.

        BDir = B[i]/scipy.linalg.norm(B[i])
        thetaBN[i] = np.arccos(np.dot(norms[i], BDir))/np.pi*180.

    return normRadDegDiff, norms, thetaBN

#takes electron Temp, ion temp,
#The vector U1 denotes the velocity of the plasma relative to the shock,
#B is the magnetic field vector that intersects the shock at a specific point
#B1 U1 are upstream values, ie further out from the shock, to be shocked as CME moves
#B2 is strentgth of magnetic field downstream or after shocked
def getflux_Cairns(Ti, Te, thetaBN, U1, B1, B2):

    #ePhi = 2kBTe(B2/B1 - 1)
    e = 1.6021766e-19 #coloumbs
    me = 9.109e-31 # mass electron kg
    mi = 1.67262e-27 #mass proton kg
    kb = 1.3807e-23 #boltzmann const
    kappa = 2.5 #kappa dist, this said to be realistic from Cairns
    c = 299792458.
    AU=149597870700. #meters
    Rs = 6.957E8 #meters
    dRs = .53/2 #degrees of Rs in sky


    #n edelPhi = 2kBTe(B2/B1 - 1) from Kuncic et al 2002
    eDelPhi = 2*kb*Te*(norm(B2)/norm(B1) - 1)

    ve = (kb*Te/me)**.5 #electron thermal speed

    #from page 3 of Cairns
    vd = norm(np.cross(-1*np.cross(U1, B1), B1)/norm(B1)**2)
    vc = vd*np.tan(thetaBN) # = v||WH


    #appendix B from Cairns
    a = vc - (ve**2)/(2*kappa*vc)*(1 + (vc/ve)**2 + (2/me*eDelPhi)/(ve**2*(norm(B2)/norm(B1) - 1))) * (np.sqrt(1 + ((1 + (vc/ve)**2)/(1 + (vc/ve)**2 + (2/me*eDelPhi)/(ve**2*(norm(B2)/norm(B1) - 1))))**(kappa+1)) - 1)

    b =  vc + (ve**2)/(2*kappa*vc)*(1 + (vc/ve)**2 + (2/me*eDelPhi)/(ve**2*(norm(B2)/norm(B1) - 1)))/(1 + ((1 + (vc/ve)**2 + (2/me*eDelPhi)/(ve**2*(norm(B2)/norm(B1) - 1)))/(1 + (vc/ve)**2 ))**(kappa+1)) *(1 - (((1 + (vc/ve)**2 + (2/me*eDelPhi)/(ve**2*(norm(B2)/norm(B1) - 1)))/(1 + (vc/ve)**2 ))**(kappa+1)) *(np.sqrt(1 + ((1 + (vc/ve)**2)/(1 + (vc/ve)**2 + (2/me*eDelPhi)/(ve**2*(norm(B2)/norm(B1) - 1))))**(kappa+1)) - 1))

    #calculate derived reflected beam PARAMETERS
    Fia = gammaFunc(kappa+1)/gammaFunc(kappa-.5)*np.pi**-.5/kappa/ve*(1 + a**2/ve**2)**(-1*kappa)
    vb = .5*(a+b)
    delvb = b - a
    nb = (b - a)*Fia

    #calculate zeta H/F from eqn 24, 25

    uc = 2.1 #typical value from Robinson Cairns 1998
    gammat = 1 + 3.*Ti/Te
    betawidth = 1/3.

    zetaF = np.exp(-1*((4*gammat*me)/(45*mi)) * ((vb)/(betawidth*delvb))**2 * (1.5*np.sqrt(mi/(gammat*me)) - vb/ve)**2 )
    zetaH = c/2./vb*np.sqrt(np.pi/6.)*betawidth*delvb/vb* (errFunc((ve*np.sqrt(3.)/c + 2./3.*np.sqrt(gammat*me/mi))/(ve*betawidth*delvb/vb**2*np.sqrt(2.))) + errFunc((ve*np.sqrt(3.)/c - 2./3.*np.sqrt(gammat*me/mi))/(ve*betawidth*delvb/vb**2*np.sqrt(2.))))

    #now move to eqn 22 23
    #the ratio of the damping rates of the product waves in the electrostatic Langmuir wave decay process
    gammaLS = (80./7.)*(ve/vb)**2*np.sqrt(mi/7./me) #Knock 2001
    phiF = 72.*np.sqrt(3.)*gammaLS*ve**3*vb*np.exp(-1*uc**2)/c**3./delvb/uc/np.sqrt(np.pi)*zetaF
    phiH = 18.*np.sqrt(3.)/5./gammat*np.sqrt(mi/gammat/me)*vb**2*ve**3/c**5*vb/delvb*zetaH

    #calculate equation 21, & 26 for brightness of each component

    delOmegaF = np.pi/4.
    delOmegaH = 2.*np.pi
    delfF = 3.*(ve/vb)**2*(delvb/vb)
    delfH = 12.*(ve/vb)**2*(delvb/vb)

    #eqn 21, volume emissivities
    # W*m^-3 sr^-1
    jF = phiF/delOmegaF*nb*me*vb**3*delvb/vb/3/1.  #change to l(r) instead of AU
    jH = phiH/delOmegaH*nb*me*vb**3*delvb/vb/3/1.  #assume l=1 on the shock.  l is the distance along the magnetic field line from point of acceleration to the point in question

    #eqn 26, get flux density, integrate over source volume j & account for propagation to Earth
    fluxF = delOmegaF/delfF*jF/AU**2
    fluxH = delOmegaH/delfH*jH/AU**2

    #print(type(fluxF))
    #print(type(delvb))
    #print(type(vb))
    #print(type(ve))
    #print(type(nb))
    #print(type(phiF))


    return fluxF, fluxH #flux density of point as seen from 1 AU


############################
# PARAMETERS TO CHANGE

##MAIN main

#create data files to save to file as well to skip later

#wind freqs
freqs1 = linspace(20., 1040., 256)
freqs2 = linspace(1075., 13825., 256)
freqs3 = linspace(14000, 25000, 23)
freqs = append(freqs1, freqs2)#when no interpolating, don't need higher things append(append(freqs1, freqs2), freqs3)


#define frequencies to show binning
newFreqs = linspace(100, 25000, 4096)

starts = logspace(log10(10.), log10(4095.), 64).astype(int)
starts[:9] = [0, 2, 4, 6, 8, 11, 14, 17, 20] #better samples at low end here
stepFreqs = newFreqs[starts]

stepFreqs = stepFreqs/1000. #to mhz



times1 = linspace(8, 36, 15).astype(int) #every 2
times2 = linspace(40, 80, 11).astype(int) #every 4
times3 = np.array([85, 90, 92, 95, 100, 105, 110, 115, 120])#, 180])
times = np.append(np.append(times1, times2), times3)


#times = [20]

#
#for datFile in ['IsoEntropy=4_20min.dat', 'IsoEntropy=4_40min.dat', 'IsoEntropy=4_60min.dat', \
#'IsoEntropy=4_80min.dat', 'IsoEntropy=4_100min.dat', 'IsoEntropy=4_120min.dat', 'CurrentSheet_20min.dat', \
#'CurrentSheet_60min.dat', 'IsoNe=3.5_20min.dat', 'IsoNe=3.5_40min.dat', 'IsoNe=3.5_60min.dat', 'IsoNe=3.5_80min.dat', 'IsoNe=3.5_100min.dat', 'IsoNe=3.5_120min.dat'] :

    #add colorbar
        #get discrete colormap
cmap = plt.get_cmap('jet', 64)
cmaplist = [cmap(i) for i in range(cmap.N)]

            # define the bins and normalize
bounds = np.append(0, stepFreqs)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

newMins = array([ 180,  480,  600,  720,  840,  960, 1080, 1200, 1320, 1440, 1560, 1680, 1800, 2160])

times = np.append(np.append(times1, times2), np.append(times3, newMins))

times = np.append(np.append(times1, times2), times3) #first 2 hours only

for time in [10, 20, 30, 40]: #times: #newMins:#[180]:#times:

    #create figure



    #calculate shock line from entropy shick projected in 2d

    #front = front of nose of shock from entropy
    #flank = one side of entropy shock
    #sheet = current sheet (own data file)
    #lobe = distributed emission, over full big density enhanced lobe (own file)
    modes = ['Front', 'Flank', 'Sheet', 'Lobe', 'Mach', 'AllSim']

    datPrefix = ['IsoEntropy=4_', 'IsoEntropy=4_', 'IsoEntropy=4_', 'isoNe=3.5_', '2005CME_iso_Machms=3_', '2005CME_interp200x3_t='] #'CurrentSheet_'

    downloadData = False
    #
    # for m in range(len(modes)):
    for m in [4]: #[0, 3]:#, 3]:
        mode = modes[m]
    # for mode in ['Front', 'Lobe']:

        #make entropy shock curve
        datFile = datPrefix[m] + str(time) +'min.dat'

        if m < 4: #diff naming convention for later 2005* files
            if time < 10:
                datFile = datPrefix[m] + '00' + str(time) +'min.dat'
            elif time < 100:
                datFile = datPrefix[m] + '0' + str(time) +'min.dat'

            elif time == 180:
                if mode == modes[0]:
                    datFile = 'IsoEntropy=4_180min.dat' #'isoEnt=4.0_3.0hr_SC.dat'
                if mode == modes[3]:
                    datFile = 'isoNe=3.5_180min.dat' #'isoNp=3.5_3.0hr_SC.dat'



        if downloadData and mode ==  modes[0]:
            #http://herot.engin.umich.edu/~chipm/files/Alex/Entropy/IsoEntropy=4_timesmin.dat
            #http://herot.engin.umich.edu/~chipm/files/Alex/Density/isoNe=3.5_
            cmd1 = 'wget http://herot.engin.umich.edu/~chipm/files/Alex/Entropy/IsoEntropy=4_' + str(time) + 'min.dat.gz'

            if time < 10:
                cmd1 = 'wget http://herot.engin.umich.edu/~chipm/files/Alex/Entropy/IsoEntropy=4_00' + str(time) + 'min.dat.gz'
            elif time < 100:
                cmd1 = 'wget http://herot.engin.umich.edu/~chipm/files/Alex/Entropy/IsoEntropy=4_0' + str(time) + 'min.dat.gz'


            returned_value = subprocess.call(cmd1, shell=True)  # returns the exit code in unix

            cmd2 = 'gunzip -f IsoEntropy=4_' + str(time) + 'min.dat'
            if time < 10:
                cmd2 = 'gunzip -f IsoEntropy=4_00' + str(time) + 'min.dat.gz'
            elif time < 100:
                cmd2 = 'gunzip -f IsoEntropy=4_0' + str(time) + 'min.dat.gz'

            returned_value = subprocess.call(cmd2, shell=True)  # returns the exit code in unix


        if downloadData and mode ==  modes[3]:
            #http://herot.engin.umich.edu/~chipm/files/Alex/Entropy/IsoEntropy=4_timesmin.dat
            #http://herot.engin.umich.edu/~chipm/files/Alex/Density/isoNe=3.5_
            cmd1 = 'wget http://herot.engin.umich.edu/~chipm/files/Alex/Density/isoNe=3.5_' + str(time) + 'min.dat.gz'

            if time < 10:
                cmd1 = 'wget http://herot.engin.umich.edu/~chipm/files/Alex/Density/isoNe=3.5_00' + str(time) + 'min.dat.gz'
            elif time < 100:
                cmd1 = 'wget http://herot.engin.umich.edu/~chipm/files/Alex/Density/isoNe=3.5_0' + str(time) + 'min.dat.gz'


            returned_value = subprocess.call(cmd1, shell=True)  # returns the exit code in unix

            cmd2 = 'gunzip isoNe=3.5_' + str(time) + 'min.dat'
            if time < 10:
                cmd2 = 'gunzip isoNe=3.5_00' + str(time) + 'min.dat.gz'
            elif time < 100:
                cmd2 = 'gunzip isoNe=3.5_0' + str(time) + 'min.dat.gz'

            returned_value = subprocess.call(cmd2, shell=True)  # returns the exit code in unix



        RsLimit = 1.2

        me = 9.109e-31 # mass electron kg
        mi = 1.67262e-27 #mass proton kg

        #
        ##make 2 plots, 2D and 3D, for every set of low-high


        lows = stepFreqs
        highs = np.append(stepFreqs[1:], 30)

        #give any axis, no need norm 1, and theta radians
        axis, theta = [0,0,1], 0.0


        ##################################


        #create rotation matrix to use later
        M0 = M(axis, theta)

        #if txt file is saved and loading from that, set false
        makeData = True

        AU=149597870700. #meters
        Rs = 6.957E8 #meters
        dRs = .53/2 #degrees of Rs in sky

        rskm = Rs*1e-3 #to km


        if makeData == True: # and (mode ==  modes[0] or mode ==  modes[3]):


            print(datFile)
            f = open(datFile)
            l = f.readlines()
            numNodes = len(l) - 1
            for i in range(len(l)):
                words = l[i].split()

                if words[0][:5] == 'Nodes':
                    nums = words[0].split('=')
                    numNodes = int(nums[1][:-2])


                if words[0] == 'DT=(SINGLE':
                    metalimit = i+1
                    break


            dataLim = metalimit + numNodes
            dataLim = min(dataLim, len(l) - 1)



            #start script
            f = open(datFile)

            allData = f.readlines()

            #seperate metadata with realness
            meta = allData[:metalimit]

            print(meta)

            restData = allData[dataLim:]

            allData = allData[metalimit:dataLim]

            lines = len(allData)

            print(allData[:20])

            ALLxs = np.zeros(lines)
            ALLys = np.zeros(lines)
            ALLzs = np.zeros(lines)
            ALLfs = np.zeros(lines)

            ALLuxs = np.zeros(lines)
            ALLuys = np.zeros(lines)
            ALLuzs = np.zeros(lines)

            ALLBxs = np.zeros(lines)
            ALLBys = np.zeros(lines)
            ALLBzs = np.zeros(lines)

            ALLI01s = np.zeros(lines)
            ALLI02s = np.zeros(lines)

            ALLNxs = np.zeros(lines)
            ALLNys = np.zeros(lines)
            ALLNzs = np.zeros(lines)

            ALLnps = np.zeros(lines)
            ALLnpRatio = np.zeros(lines)
            ALLps = np.zeros(lines)
            ALLrhos = np.zeros(lines)

            ALLent = np.zeros(lines)
            ALLentRatio = np.zeros(lines)

            ALLTis = np.zeros(lines)
            ALLTes = np.zeros(lines)

            ALLcs = np.zeros(lines)
            ALLca = np.zeros(lines)
            ALLcms = np.zeros(lines)
            ALLV = np.zeros(lines)
            ALLMms = np.zeros(lines)
            ALLrRs = np.zeros(lines)
            ALLMmscme = np.zeros(lines)

            e = 1.6021766e-19 #coloumbs
            me = 9.109e-31 # mass electron kg
            e0 = 8.85418781e-12 #Farads / meter permittivity of free space

            #now 17 vars
            #x,y,z,density g/cc, ux, uy, uz, t_ion, t_e, Bx, By, Bz, I01, I02, p (ressure?), np ratio, PF MHz, (Nxyz)
            for i in range(lines):
                # if i == 237424:
                #     continue
                s = allData[i].split()
                if len(s) < 10:
                    continue
                ALLxs[i] = float(s[0])
                ALLys[i] = float(s[1])
                ALLzs[i] = float(s[2])
                ALLrhos[i] = float(s[3]) # grams per cc
                ALLuxs[i] = float(s[4])
                ALLuys[i] = float(s[5])
                ALLuzs[i] = float(s[6])
                ALLTis[i] = float(s[7])
                ALLTes[i] = float(s[8])
                ALLBxs[i] = float(s[9])
                ALLBys[i] = float(s[10])
                ALLBzs[i] = float(s[11])
                ALLI01s[i] = float(s[12])
                ALLI02s[i] = float(s[13])
                ALLps[i] = float(s[14]) #pressure
                ALLnpRatio[i] = float(s[15]) #np ratio in SC data only, in IH data is dt s
                ALLent[i] = float(s[16])
                # ALLentRatio[i] = float(s[16])

                ALLfs[i] = float(s[17])



                ALLnps[i] = ALLrhos[i]/1000./(me+mi) #*100.**3 # to kilograms to ne per cc
                # if time < 180:
                    # ALLfs[i] = np.sqrt((ALLnps[i]*ALLnpRatio[i])*100.**3*e**2/me/e0)/2/np.pi/1e6 #gives MHz plasma freq

                ALLfs[i] = np.sqrt((ALLnps[i])*100.**3*e**2/me/e0)/2/np.pi/1e6 #gives MHz plasma freq

                if m >= 4: #mach speed and AllSim
                    ALLfs[i] = float(s[15]) #"PF MHz"
                    ALLent[i] = float(s[16]) #"Entropy"
                    ALLcs[i] = float(s[17])
                    ALLca[i] = float(s[18])
                    ALLcms[i] = float(s[19])
                    ALLV[i] = float(s[20])
                    ALLnpRatio[i] = float(s[21])
                    ALLentRatio[i] = float(s[22])
                    ALLMms[i] = float(s[23])
                    if m == 4: #only mach speed
                        ALLrRs[i] = float(s[24])
                        ALLMmscme[i] = float(s[25])




                # ALLfs[i] = float(s[16])
                # ALLnps[i] = float(s[15])


        #
        #        if datFile==dats[3]:
        #            ALLNxs[i] = float(s[17])
        #            ALLNys[i] = float(s[18])
        #            ALLNzs[i] = float(s[19])
        #
        #        if datFile == dats[5]:
        #            ALLent[i] = float(s[16])
        #            ALLentRatio[i] = float(s[17])
        #            ALLfs[i] = float(s[18])



            f.close()

            print('done reading file')

            t = Time('2005-05-13T16:57:00', format='isot', scale='utc')
            # t0 = Time('1853-11-09T00:00:00', format='isot', scale='utc')

            t = t + time*(1./24/60)


            #use astropy to convert XYZ to GSE getCoordinates
            #get HEEQ coordinates, X is Sun Earth line, Z is Solar north pole.  heliocentric
            if time <= 180:
                #SC simulation is in HGR (heliographic_carrington in sunpy)
                #for loop over ALLxs

                for pt in range(len(ALLxs)):
                    c = SkyCoord(CartesianRepresentation(x=ALLxs[pt]*rskm*u.km, y=ALLys[pt]*rskm*u.km, z=ALLzs[pt]*rskm*u.km), frame="heliographic_carrington", obstime=t, observer='earth')
                    # c = SkyCoord(CartesianRepresentation(x=1.*u.km, y=1.*u.km, z=1.*u.km), frame="heliographic_carrington", obstime=t)
                    # sc = SkyCoord(CartesianRepresentation(0*u.km, 45*u.km, 2*u.km), obstime="2011/01/05T00:00:50", frame="heliographic_carrington")

                    csh = c.transform_to(frame="heliographic_stonyhurst")

                    #get HEEQ frame
                    xyzh = csh.represent_as(CartesianRepresentation)
                    ALLxs[pt] = xyzh.x.value/rskm #back to Rs from km
                    ALLys[pt] = xyzh.y.value/rskm
                    ALLzs[pt] = xyzh.z.value/rskm

                    #now for U and B
                    c = SkyCoord(CartesianRepresentation(x=ALLuxs[pt]*rskm*u.km, y=ALLuys[pt]*rskm*u.km, z=ALLuzs[pt]*rskm*u.km), frame="heliographic_carrington", obstime=t)
                    # c = SkyCoord(CartesianRepresentation(x=1.*u.km, y=1.*u.km, z=1.*u.km), frame="heliographic_carrington", obstime=t)
                    # sc = SkyCoord(CartesianRepresentation(0*u.km, 45*u.km, 2*u.km), obstime="2011/01/05T00:00:50", frame="heliographic_carrington")

                    csh = c.transform_to(frame="heliographic_stonyhurst")

                    #get HEEQ frame
                    xyzh = csh.represent_as(CartesianRepresentation)
                    ALLuxs[pt] = xyzh.x.value/rskm #back to Rs from km
                    ALLuys[pt] = xyzh.y.value/rskm
                    ALLuzs[pt] = xyzh.z.value/rskm

                    #B now
                    c = SkyCoord(CartesianRepresentation(x=ALLBxs[pt]*rskm*u.km, y=ALLBys[pt]*rskm*u.km, z=ALLBzs[pt]*rskm*u.km), frame="heliographic_carrington", obstime=t)
                    # c = SkyCoord(CartesianRepresentation(x=1.*u.km, y=1.*u.km, z=1.*u.km), frame="heliographic_carrington", obstime=t)
                    # sc = SkyCoord(CartesianRepresentation(0*u.km, 45*u.km, 2*u.km), obstime="2011/01/05T00:00:50", frame="heliographic_carrington")

                    csh = c.transform_to(frame="heliographic_stonyhurst")

                    #get HEEQ frame
                    xyzh = csh.represent_as(CartesianRepresentation)
                    ALLBxs[pt] = xyzh.x.value/rskm #back to Rs from km
                    ALLBys[pt] = xyzh.y.value/rskm
                    ALLBzs[pt] = xyzh.z.value/rskm


            else:
                #IH simulation after 20 or 24 Rs in HGI coord system (heliocentricinertial in astropy)
                for pt in range(len(ALLxs)):
                    c = SkyCoord(CartesianRepresentation(x=ALLxs[pt]*rskm*u.km, y=ALLys[pt]*rskm*u.km, z=ALLzs[pt]*rskm*u.km), frame="heliocentricinertial", obstime=t)

                    csh = c.transform_to(frame="heliographic_stonyhurst")

                    #get HEEQ frame
                    xyzh = csh.represent_as(CartesianRepresentation)
                    ALLxs[pt] = xyzh.x.value/rskm #back to Rs from km
                    ALLys[pt] = xyzh.y.value/rskm
                    ALLzs[pt] = xyzh.z.value/rskm

                    #now for U and B
                    c = SkyCoord(CartesianRepresentation(x=ALLuxs[pt]*rskm*u.km, y=ALLuys[pt]*rskm*u.km, z=ALLuzs[pt]*rskm*u.km), frame="heliographic_carrington", obstime=t)
                    # c = SkyCoord(CartesianRepresentation(x=1.*u.km, y=1.*u.km, z=1.*u.km), frame="heliographic_carrington", obstime=t)
                    # sc = SkyCoord(CartesianRepresentation(0*u.km, 45*u.km, 2*u.km), obstime="2011/01/05T00:00:50", frame="heliographic_carrington")

                    csh = c.transform_to(frame="heliographic_stonyhurst")

                    #get HEEQ frame
                    xyzh = csh.represent_as(CartesianRepresentation)
                    ALLuxs[pt] = xyzh.x.value/rskm #back to Rs from km
                    ALLuys[pt] = xyzh.y.value/rskm
                    ALLuzs[pt] = xyzh.z.value/rskm

                    #B now
                    c = SkyCoord(CartesianRepresentation(x=ALLBxs[pt]*rskm*u.km, y=ALLBys[pt]*rskm*u.km, z=ALLBzs[pt]*rskm*u.km), frame="heliographic_carrington", obstime=t)
                    # c = SkyCoord(CartesianRepresentation(x=1.*u.km, y=1.*u.km, z=1.*u.km), frame="heliographic_carrington", obstime=t)
                    # sc = SkyCoord(CartesianRepresentation(0*u.km, 45*u.km, 2*u.km), obstime="2011/01/05T00:00:50", frame="heliographic_carrington")

                    csh = c.transform_to(frame="heliographic_stonyhurst")

                    #get HEEQ frame
                    xyzh = csh.represent_as(CartesianRepresentation)
                    ALLBxs[pt] = xyzh.x.value/rskm #back to Rs from km
                    ALLBys[pt] = xyzh.y.value/rskm
                    ALLBzs[pt] = xyzh.z.value/rskm


                ##################################coordsys trans done
            #compute radius
            ALLrs = np.sqrt(ALLxs**2 + ALLys**2 + ALLzs**2)
            r2d = np.sqrt(ALLxs**2 + ALLys**2)


            #restrict to outer shell
            smallR = np.where(ALLrs < RsLimit)[0]
            # for 'IsoEntropy=4_20min.dat'
            #smallR = np.where(np.logical_or( np.logical_or(r2d > 1.7/dRs ,np.logical_or(ALLxs > (1.15/dRs), ALLys > (1.55/dRs))), np.logical_or(np.logical_or(ALLrs < RsLimit, ALLrs > 7.),  np.logical_or(np.logical_or(ALLxs > (1.18/dRs), np.logical_and(ALLzs > (0.3/dRs),ALLxs > (1.1/dRs))), np.logical_and(ALLzs > (0.5/dRs),ALLxs > (1.05/dRs))))))[0]

            goodR = np.delete(ALLrs, smallR)



            print('Culling points inside radial cutoff')

            xs = np.delete(ALLxs, smallR)
            ys = np.delete(ALLys, smallR)
            zs = np.delete(ALLzs, smallR)
            fs = np.delete(ALLfs, smallR)

            uxs = np.delete(ALLuxs, smallR)
            uys = np.delete(ALLuys, smallR)
            uzs = np.delete(ALLuzs, smallR)

            Bxs = np.delete(ALLBxs, smallR)
            Bys = np.delete(ALLBys, smallR)
            Bzs = np.delete(ALLBzs, smallR)

            Nxs = np.delete(ALLNxs, smallR)
            Nys = np.delete(ALLNys, smallR)
            Nzs = np.delete(ALLNzs, smallR)

            nps = np.delete(ALLnps, smallR)
            #nps = np.delete(ALLnpRatio, smallR)

            ps = np.delete(ALLps, smallR)
            rhos = np.delete(ALLrhos, smallR)

            ent = np.delete(ALLent, smallR)
            entRatio = np.delete(ALLentRatio, smallR)

            Tis = np.delete(ALLTis, smallR)
            Tes = np.delete(ALLTes, smallR)

            cs = np.delete(ALLcs, smallR)
            ca = np.delete(ALLca, smallR)
            cms = np.delete(ALLcms, smallR)
            npRatio = np.delete(ALLnpRatio, smallR)
            V = np.delete(ALLV, smallR)
            Mms = np.delete(ALLMms, smallR)
            rRs = np.delete(ALLrRs, smallR)
            Mmscme = np.delete(ALLMmscme, smallR)



            #done with preliminary cuts


            entropys = ps/(rhos)**(5./3.)
            #delete duplicate position points, keep higher freq
            pos = np.array([xs, ys, zs]).T
            N = len(xs)


            if m != 5: #don't do slow calc with big files
                nbrs = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(pos)
                distances, indices = nbrs.kneighbors(pos)

                dupes = np.zeros(N)
                dupeInds = []


                for i in range(N):
                    #has a neighbour not itself that is on top of it, and we havent gotten its partner yet
                    if distances[i][1] == 0. and dupes[indices[i][1]] == 0:
                        dupes[indices[i][0]] = 1.
                        dupes[indices[i][1]] = 1.
                        dupeInds.append((indices[i][0], indices[i][1])) # this would make len(lodupes) == len(hidupes) < N by 1 each instance

                    if distances[i][1] > 0.:
                        dupes[indices[i][0]] = 1.
                        #dupes[indices[i][1]] = 1.
                        dupeInds.append((indices[i][0], indices[i][0])) #if this happens every time, N = len(hidupes) = len(lodupes), so no dupes




                #find the lower frequency dupes
                lodupes = []
                hidupes = []
                for i in range(len(dupeInds)):
                    if fs[dupeInds[i][0]] > fs[dupeInds[i][1]]:
                        lodupes.append(dupeInds[i][1])
                        hidupes.append(dupeInds[i][0])


                    else:
                        lodupes.append(dupeInds[i][0])
                        hidupes.append(dupeInds[i][1])


                print('Culling duplicate points')
                print(N)
                print(len(lodupes))
                print(len(hidupes))

                lofs = fs[lodupes]
                louxs = uxs[lodupes]
                louys = uys[lodupes]
                louzs = uzs[lodupes]
                loBxs = Bxs[lodupes]
                loBys = Bys[lodupes]
                loBzs = Bzs[lodupes]
                loxs = xs[lodupes]
                loys = ys[lodupes]
                lozs = zs[lodupes]

                lorhos = rhos[lodupes]


                #cut those dupes
                xs = xs[hidupes] #np.delete(xs, lodupes)
                ys = ys[hidupes] #np.delete(ys, lodupes)
                zs = zs[hidupes] #np.delete(zs, lodupes)
                fs = fs[hidupes] #np.delete(fs, lodupes)

                uxs = uxs[hidupes] #np.delete(uxs, lodupes)
                uys = uys[hidupes] #np.delete(uys, lodupes)
                uzs = uzs[hidupes] #np.delete(uzs, lodupes)

                Bxs = Bxs[hidupes] #np.delete(Bxs, lodupes)
                Bys = Bys[hidupes] #np.delete(Bys, lodupes)
                Bzs = Bzs[hidupes] #np.delete(Bzs, lodupes)


                Nxs = Nxs[hidupes] #np.delete(Nxs, lodupes)
                Nys = Nys[hidupes] #np.delete(Nys, lodupes)
                Nzs = Nzs[hidupes] #np.delete(Nzs, lodupes)

                nps = nps[hidupes] #np.delete(nps, lodupes) upstream, about to be shocked normal SW
                ps = ps[hidupes] #np.delete(ps, lodupes)
                rhos = rhos[hidupes] #np.delete(rhos, lodupes)

                ent = ent[hidupes] #np.delete(ent, lodupes)
                entRatio = entRatio[hidupes] #np.delete(entRatio, lodupes)

                cs = cs[hidupes]
                ca = ca[hidupes]
                cms = cms[hidupes]
                npRatio = npRatio[hidupes]
                V = V[hidupes]
                Mms = Mms[hidupes]
                rRs = rRs[hidupes]
                Mmscme = Mmscme[hidupes]


            else:
                lofs = fs
                louxs = uxs
                louys = uys
                louzs = uzs
                loBxs = Bxs
                loBys = Bys
                loBzs = Bzs
                loxs = xs
                loys = ys
                lozs = zs


            #Nx3 position vector
            pos = np.array([xs, ys, zs]).T
            N = len(xs)

            print(str(N) + ' total points')

            #norm
            Ns = np.array([Nxs, Nys, Nzs]).T

            #print(pos[9], pos[10], 'check1')

            velos = np.array([uxs, uys, uzs]).T

            magField = np.array([Bxs, Bys, Bzs]).T

            lovelos = np.array([louxs, louys, louzs]).T

            lomagField = np.array([loBxs, loBys, loBzs]).T

            ### Now create view from Earth

            #apply rotation matrix to positions if needed

            pos = dot(M0, pos.T).T
            velos = dot(M0, velos.T).T
            magField = dot(M0, magField.T).T
            Ns = dot(M0, Ns.T).T

            lovelos = dot(M0, lovelos.T).T
            lomagField = dot(M0, lomagField.T).T

            NDiff = np.zeros(N)

            print('calculating norms and thetaBN')
            normRadDegDiff = np.zeros(N)
            norms = np.zeros(N)
            thetaBN = np.zeros(N)

            #For calculating all thetaRN at once
            if m!= 5:
                normRadDegDiff, norms, thetaBN = getThetaRN(pos, N, lomagField)
            # normRadDegDiff = np.zeros(N)
            # norms = np.zeros([N, 3])
            # thetaBN= np.zeros(N)
        #
            Ns = norms


            for i in range(N):
                radDir = pos[i]/scipy.linalg.norm(pos[i])
                NDiff[i] = np.arccos(np.dot(Ns[i], radDir))/np.pi*180.

                if NDiff[i] > 90:
                    Ns[i] *= -1
                    NDiff[i] = np.arccos(np.dot(Ns[i], radDir))/np.pi*180.






            ### Going forward, assume x axis is Sun->Earth, project to y-z plane of sky

            AU=149597870700. #meters
            Rs = 6.957E8 #meters
            dRs = .53/2 #degrees of Rs in sky




            #now calculate hoffmann teller frame velocity, use norms or ASNmx3

            HTVelo = np.zeros([N, 3])
            fluxF = np.zeros(N)
            fluxH = np.zeros(N)
            if m!= 5:
                for i in range(N):
                    if m != 4:
                        numer = np.cross(Ns[i], np.cross(velos[i] - lovelos[i], lomagField[i]))
                    else:
                        veloDiff = (V[i] - scipy.linalg.norm(velos[i])) * velos[i] #alt calc using mean event speed V, and assuming smaller velos is upstream
                        numer = np.cross(Ns[i], np.cross(veloDiff, lomagField[i]))

                    denom = np.dot(lomagField[i], Ns[i])
                    # fluxF[i], fluxH[i] = getflux_Cairns(Tis[i], Tes[i], thetaBN[i], velos[i] - lovelos[i], lomagField[i], magField[i])
                    HTVelo[i] = numer/denom
            #
            absHTVelo = np.apply_along_axis(linalg.norm, 1, HTVelo)#norm(HTVelo, axis=1)



            #now remove any lower frequency points behind higher frequency ones.

            #sort [pos,freq] by x direction,



            dataArr = [(x, y, z, ux, uy, uz, Bx, By, Bz, f, n1, n2, n3, fv1, fv2, fv3, ft, tbn, Nx, Ny, Nz, NDiff, p, npsingle, rho, ent, entrat, fluxf, fluxh, CS, CA, CMS, VV, nprat, MMS, RRS, MMSCME) \
            for x, y, z, ux, uy, uz, Bx, By, Bz, f, n1, n2, n3, fv1, fv2, fv3, ft, tbn, Nx, Ny, Nz, NDiff, p, npsingle, rho, ent, entrat, fluxf, fluxh, CS, CA, CMS, VV, nprat, MMS, RRS, MMSCME in \
            zip(pos[:, 0], pos[:, 1], pos[:, 2], velos[:, 0], velos[:, 1], velos[:, 2], \
            lomagField[:, 0], lomagField[:, 1], lomagField[:, 2], lofs, norms[:, 0], norms[:, 1], norms[:, 2], HTVelo[:, 0], HTVelo[:, 1], HTVelo[:, 2], absHTVelo, thetaBN, Ns[:, 0], Ns[:, 1], Ns[:, 2], NDiff, ps, nps, rhos, entropys, entRatio, fluxF, fluxH, cs, ca, cms, V, npRatio, Mms, rRs, Mmscme)]


            data=np.array(sorted(dataArr))

            json.dump(data.tolist(), open(datFile+"dataUNCUTTBN.txt", 'w'))

            #don't cut data for allsim files
            if m == 5:
                continue

            #project to 2D y-z POS
            deg2D = data[:, 1:3]

            N = len(data[:, 0])

            obscureDist = .001 #if within this POS angular dist and higher freq in front of lower, then obscure lower

            print('Obscuring low freq LOS points')

            #find neighbours in 2D projected space to find LOS buddy points
            #algorithm kd_tree for fast, or brute for slow
            #http://scikit-learn.org/stable/modules/neighbors.html
            nbrs = NearestNeighbors(radius=obscureDist, algorithm='kd_tree').fit(deg2D)
            distances, indices = nbrs.radius_neighbors(deg2D)

            #indices to remove after initial pass
            dead =  np.zeros(N)

            deadInds = []

            #move backwards through x, assuming it grows towards Earth, so we're closest to hi x
            for i in range(N-1, -1, -1):

                #if obscured, something else obscured everything this would've, skip
                if dead[i] == 1.:
                        continue

                LOSbuds = indices[i]

                for j in range(len(LOSbuds)):

                    #freq of buddy less than freq of i, obscured, 9 is ind into freq col of data
                    if data[LOSbuds[j]][9] < data[i][9]:
                        if dead[LOSbuds[j]] == 1.:
                            continue

                        dead[LOSbuds[j]] = 1.
                        deadInds.append(LOSbuds[j])


            #delete obscured points
            visData = np.delete(data, deadInds, axis=0)


            print(str(len(deadInds)) + ' total lower frequency points obscured in this projection')

            print(str(len(visData[:, 0])) + ' total unobscured points')


            json.dump(visData.tolist(), open(datFile+"dataTBN.txt", 'w'))


        else:
            if m != 5:
                visDataArr =  json.load(open(datFile+"dataTBN.txt", 'r'))
            else:
                visDataArr =  json.load(open(datFile+"dataUNCUTTBN.txt", 'r'))

            visData = np.array(sorted(visDataArr))



        #do not make trad figures of all data
        if m == 5:
            #make unique figures for histograms of different values
            clf()
            fig = figure()
            h = hist(visData[:,26], bins = 30)
            title('Histogram of Entropy Ratio for Time %s'%(str(time)))#, fontsize=25)
            # xlabel(r'Entropy Ratio')#, fontsize=25)
            # ylabel(r'Data Cell Counts')#, fontsize=25)
            savefig('entRatHist%s.png'%(str(time)))
            clf()

            clf()
            fig = figure()
            h = hist(visData[:,33], bins = 30)
            title('Histogram of Np Ratio for Time %s'%(str(time)))#, fontsize=25)
            # xlabel(r'Entropy Ratio')#, fontsize=25)
            # ylabel(r'Data Cell Counts')#, fontsize=25)
            savefig('npRatHist%s.png'%(str(time)))
            clf()

            clf()
            fig = figure()
            h = hist(visData[:,34], bins = 30)
            title('Histogram of Magnetosonic Mach # for Time %s'%(str(time)))#, fontsize=25)
            # xlabel(r'Entropy Ratio')#, fontsize=25)
            # ylabel(r'Data Cell Counts')#, fontsize=25)
            savefig('magMachHist%s.png'%(str(time)))
            clf()

            clf()
            fig = figure()
            h = hist(visData[:,36], bins = 30)
            title('Histogram of CME Magnetosonic Mach # for Time %s'%(str(time)))#, fontsize=25)
            # xlabel(r'Entropy Ratio')#, fontsize=25)
            # ylabel(r'Data Cell Counts')#, fontsize=25)
            savefig('magMachCMEHist%s.png'%(str(time)))
            clf()
            continue

        #all below is making snapshot of data, no more multiprocessing
        # continue

        #make thetaBN between 0-90 only, makes graphs easier to read
        lengthVis = shape(visData)[0]
        for k in range(lengthVis):
            if visData[k][17] > 90.:
                visData[k][17] = 180. - visData[k][17]

        newFreqs = linspace(100, 25000, 4096)

        starts = starts = logspace(log10(10.), log10(4095.), 64).astype(int)
        starts[:9] = [0, 2, 4, 6, 8, 11, 14, 17, 20] #better samples at low end here
        stepFreqs = newFreqs[starts]

        stepFreqs = stepFreqs/1000. #to mhz



        cmap = plt.get_cmap('jet', 64)
        cmaplist = [cmap(i) for i in range(cmap.N)]

            # define the bins and normalize
        bounds = np.append(0, stepFreqs)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)



            #find color of each visible point according to other parameter eg frequency or thetaBN
        #16 is abs HT velo, 9 is freq, norm 10-12, HT 13-15, 16 is abs(HT) 17 is thetaBN, #25 entropy #27/28 are Ciarns flux fund and harmonic
        colorStream = 9#17

        RS = np.sqrt(visData[:, 0]**2 +visData[:, 1]**2 + visData[:, 2]**2)

        varColor = visData[:, colorStream]
        varRes = 64
        varLow = min(varColor) #min freq 50000#
        varHigh = max(varColor) #max freq 1000000. #


        print(varLow, varHigh)

        if colorStream == 16: #avoid outliers in HT
            varHigh = 1000.

        colors = cm.rainbow(np.linspace(0, 1, varRes+1))
        cbounds = linspace(varLow, varHigh, varRes)
        cnorm = mpl.colors.BoundaryNorm(cbounds, varRes)

        #affine transformation to get closest integer ind into colors from colorVar freq etc
        colorFactor = (varHigh - varLow)/varRes

        if colorFactor == 0.:
            colorFactor = 1.

        colorInds = np.round((varColor - varLow)/colorFactor)


        #plotcolors = []
        #for i in range(len(varColor)):
        #        if int(colorInds[i]) > varRes:
        #            plotcolors.append(colors[varRes])
        #
        #	else:
        #	    plotcolors.append(colors[int(colorInds[i])])


        #project to 2D y-z POS
        deg2D = visData[:, 1:3]

        #get associated visible, sorted f vals
        fs = visData[:, 9]

        #associated shock norm diff with radial
        #normDiffs = visData[:, 4]


        #2.2 to 3.7 MHZ has nice mid belt
        lows = [0.01]
        highs = [25.]

        for freqlo, freqhi in zip(lows, highs):
            # freqRangeInds = np.where(np.logical_and(fs > freqlo, fs < freqhi))[0]

            # freqRangeInds = np.where(np.logical_and(varColor >= varLow, varColor <= varHigh))[0]
            freqRangeInds = np.where(np.logical_and(np.logical_and(fs > freqlo, fs < freqhi), np.logical_and(varColor >= varLow, varColor <= varHigh)))[0]


            numPts = len(freqRangeInds)

            print(str(numPts) + ' points plotted after frequency trimming')

            freqColors = []
            for j in range(numPts):
                 # li = bisect.bisect_left(stepFreqs, visData[:,9][freqRangeInds[j]])
                 #
                 # freqColors.append(cmap(li))
    #            if smallR[li] == nbInd: #then this neighbour is in the cuts we're correcting, move on
    #                pass

                #old general code for relative colorscale for any variable
               if int(colorInds[freqRangeInds[j]]) > varRes:
                   freqColors.append(colors[varRes])
               elif int(colorInds[freqRangeInds[j]]) < 0:
                   freqColors.append(colors[0])
               else:
                   freqColors.append(colors[int(colorInds[freqRangeInds[j]])])
        #

            fig1 = plt.figure()
            ax = fig1.add_subplot(111)

            fig1.subplots_adjust(right=0.8)
            cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
            # cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, spacing='uniform', ticks=bounds[::4], boundaries=bounds, format='%2.1f')
            cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=cnorm, spacing='uniform', ticks=cbounds[::4], boundaries=cbounds, format='%2.1f')


            # cbar_ax.set_ylabel('64 Observing Frequencies (MHz)', size=12)
            # cbar_ax.set_ylabel(r'$\Theta_{BN}$', size=12)

            sun = Ellipse(xy=(0, 0),
                        width=2.0, height=2.0,# figure = fig,
                        angle=0., fill = False)
            sun.set_facecolor('y')
            #ell.set_hatch('x')
            ax.add_artist(sun)

            p1=ax.scatter(deg2D[:,0][freqRangeInds], deg2D[:, 1][freqRangeInds], c=freqColors)

            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            ax.set_xlabel('X Solar Radii')
            ax.set_ylabel('Y Solar Radii')




            #ax.set_xlabel('X Degrees from Sun')
            #ax.set_ylabel('Y Degrees from Sun')
            ax.set_title(str(freqlo) + ' - ' + str(freqhi) + ' MHz, rotated about z axis ' + str(theta) + ' radians\n' + str(numPts) + ' points plotted for '+datFile)
            #plt.axis([-axSize, axSize, -axSize, axSize])


            #m = cm.ScalarMappable(cmap=cm.rainbow, norm = p1.norm)
            #m.set_array(varColor[freqRangeInds])
            #cbar=fig1.colorbar(m)
            cbar_ax.set_ylabel('Frequency (MHz)')
            #cbar.ax.set_ylabel('HT Frame Velocity')
            #cbar.ax.set_ylabel(r'$\Theta_{BN}$')
            #cbar.ax.set_ylabel('Entropy')
            #cbar.ax.set_ylabel('Cairns Fundamental Flux')

            plt.show()

            fileName = 'Entropy_2D%3f-%3fMHz%s.png'%(freqlo, freqhi, datFile)
            fileName = '2D'+str(freqlo) + '-' + str(freqhi) + 'MHz'+str(datFile)+'.png'

            fig1.savefig(fileName, bbox_inches='tight', pad_inches=0.0)

            plt.close()



            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            #add spherical sun
            # draw sphere
            su, sv = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            sx = np.cos(su)*np.sin(sv)
            sy = np.sin(su)*np.sin(sv)
            sz = np.cos(sv)
            ax.plot_surface(sx, sy, sz, color="y")

            #[freqRangeInds]

            skip = 10

            # p=ax.scatter(visData[:,0][freqRangeInds[::skip]], visData[:,1][freqRangeInds[::skip]], visData[:,2][freqRangeInds[::skip]], color=freqColors[::skip])
            p=ax.scatter(visData[:,0][freqRangeInds[::skip]], visData[:,1][freqRangeInds[::skip]], visData[:,2][freqRangeInds[::skip]], color=freqColors[::skip])


            # ax.set_xlim(-3, 20)
            # ax.set_ylim(-3, 20)
            # ax.set_zlim(-3, 20)

            ax.set_xlim(-2, 15)
            ax.set_ylim(-15, 15)
            ax.set_zlim(-15, 10)

            #plot vector norm arrows
            ax.quiver(visData[:,0][freqRangeInds[::skip]], visData[:,1][freqRangeInds[::skip]], visData[:,2][freqRangeInds[::skip]], visData[:,10][freqRangeInds[::skip]], visData[:,11][freqRangeInds[::skip]], visData[:,12][freqRangeInds[::skip]], length = .05, pivot='tail', color='g')
            ##plot B mag field lines upstream
            Bnorms = np.sqrt(visData[:,6][freqRangeInds[::skip]]**2 + visData[:,7][freqRangeInds[::skip]]**2 + visData[:,8][freqRangeInds[::skip]]**2)
            ax.quiver(visData[:,0][freqRangeInds[::skip]], visData[:,1][freqRangeInds[::skip]], visData[:,2][freqRangeInds[::skip]], visData[:,6][freqRangeInds[::skip]]/Bnorms, visData[:,7][freqRangeInds[::skip]]/Bnorms, visData[:,8][freqRangeInds[::skip]]/Bnorms, length = .05, pivot='tail') #.04 length old


            ax.set_xlabel('X Solar Radii')
            ax.set_ylabel('Y Solar Radii')
            ax.set_zlabel('Z Solar Radii')

            plt.title(str(freqlo) + ' - ' + str(freqhi) + ' MHz, rotated about z axis ' + str(theta) + ' radians\n' + str(numPts) + ' points plotted for ' +datFile)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, spacing='uniform', ticks=bounds[::4], boundaries=bounds, format='%2.1f')
            cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=cnorm, spacing='uniform', ticks=cbounds[::4], boundaries=cbounds, format='%2.1f')
            # cbar_ax.set_ylabel('64 Observing Frequencies (MHz)', size=12)


        #
        #    m = cm.ScalarMappable(cmap=cm.rainbow, norm = p.norm)
        #    m.set_array(varColor[freqRangeInds])
        #    cbar=fig.colorbar(m)
            cbar_ax.set_ylabel('Frequency (MHz)')
            #cbar.ax.set_ylabel('HT Frame Velocity')
            # cbar_ax.set_ylabel(r'$\Theta_{BN}$', size=12)
            #cbar.ax.set_ylabel('Entropy')
            #cbar.ax.set_ylabel('Cairns Fundamental Flux')

            plt.show()


            # fileName = 'Entropy_3D%3f-%3fMHz%s.png'%(freqlo, freqhi, datFile)
            fileName = '3D'+str(freqlo) + '-' + str(freqhi) + 'MHz'+str(datFile)+'.png'

            fig.savefig(fileName, bbox_inches='tight', pad_inches=0.0)
            plt.close()


#
# times1 = linspace(8, 36, 15).astype(int) #every 2
# times2 = linspace(40, 80, 11).astype(int) #every 4
# times3 = np.array([85, 90, 92, 95, 100, 105, 110, 115, 120])
# times = np.append(np.append(times1, times2), times3)
#
# datPrefix = ['IsoEntropy=4_', 'IsoEntropy=4_', 'IsoEntropy=4_', 'isoNe=3.5_'] #CurrentSheet_
#
# entDatFiles = ''
#
# for filet in ['2D0.1-25.0MHz', '3D0.1-25.0MHz']:
#
#
#
#     for datPrefix in ['IsoEntropy=4_', 'isoNe=3.5_'] :
#
#         fileList = ''
#
#         for t in times:
#
#             #make entropy shock curve
#             datFile = datPrefix + str(t) +'min.dat'
#
#             if t < 10:
#                 datFile = datPrefix + '00' + str(t) +'min.dat'
#             elif t < 100:
#                 datFile = datPrefix + '0' + str(t) +'min.dat'
#
#             fileList += filet+datFile+'.png '
#
#         returned_value = subprocess.call('convert -delay 30 -loop 1 '+ fileList+' '+ filet+ datPrefix+'.gif', shell=True)

#fileList = ''
#for t in times:
#
#    #make entropy shock curve
#    imfile = 'PreRadio4Panel'+str(t)+'.png '
#
#
#    fileList += imfile
#
#returned_value = subprocess.call('convert -delay 15 -loop 0 '+ fileList+' ' + 'PreRadio4Panel.gif', shell=True)
