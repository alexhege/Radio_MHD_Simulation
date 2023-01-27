#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''#Alex Hegedus
#https://solar-radio.gsfc.nasa.gov/wind/one_minute_doc.html
#all arrays are 256x1441
#All values are in terms of ratio to background and the background values are listed in position 1441 in microvolts per root Hz.
#create frequency selection figure from Wind data

This script is a main workhorse, it creates a synthetic spectra for a set of data cuts, and scores its similarity with the stencil
'''

import matplotlib
matplotlib.use('Agg')

from pylab import *
from scipy.io import readsav

import subprocess

from matplotlib.colors import LogNorm

import matplotlib.colors as colors

from scipy.stats import norm as normVar

# from scipy import interpolate


from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from sklearn.neighbors import NearestNeighbors


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import matplotlib.image as plimg
import scipy.misc
# from scipy.io import readsav

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

from matplotlib.patches import Ellipse



def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i], i
    raise ValueError


def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1], i
    raise ValueError



def my_interp(X, Y, Z, x, y, spn=5):
    xs,ys = map(np.array,(x, y))
    z = np.zeros(xs.shape)
    sx, sy = xs.shape
    for i,(x, y) in enumerate(zip(xs.flatten(),ys.flatten())):
        #print(str(np.shape(X)) + str(np.shape(x)))
        #print(x, y, i, xs, ys)
        if i%1600==0:
            print(i)
        # get the indices of the nearest X, Y
        xi = np.argmin(np.abs(X[0,:]-x))
        yi = np.argmin(np.abs(Y[:,0]-y))
        xlo = max(xi-spn, 0)
        ylo = max(yi-spn, 0)
        xhi = min(xi+spn, X[0,:].size)
        yhi = min(yi+spn, Y[:,0].size)
        # make slices of X, Y,Z that are only a few items wide
        nX = X[ylo:yhi, xlo:xhi]
        nY = Y[ylo:yhi, xlo:xhi]
        nZ = Z[ylo:yhi, xlo:xhi]
        # print(str(xlo) + ' ' + str(xhi) + ' ' + str(xi) + ' ' + str(ylo) + ' ' + str(yhi) + ' ' + str(yi))
        # print(str(np.shape(nX)) + str(np.shape(nY)) +str(np.shape(nZ)))
        intp = interpolate.interp2d(nX, nY, nZ, kind='linear', fill_value=0.)
        zi = i/sy
        zj = i - sy*zi
        z[zi, zj] = intp(x, y)[0]
    return z



#a be the unit vector along axis, i.e. a = axis/norm(axis)
#and A = I × a be the skew-symmetric matrix associated to a, i.e. the cross product of the identity matrix with a
#then M = exp(θ A) is the rotation matrix.
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

        radDir = pos[i]/norm(pos[i])

        totalCross = 0

        chooseTwo = findsubsets(inds, 2)
        manyChoose = len(chooseTwo)

        #compute normal derivative from each of 3 triangles made with center point
        for j in range(manyChoose):
            #choose 2 from set of 3, and add in point of interest, define triangle
            sub1, sub2 = chooseTwo.pop()

            vect1 = pos[sub1] - pos[i]
            vect2 = pos[sub2] - pos[i]

            cross = np.cross(vect1, vect2)
            if norm(cross) == 0.:
                # print(i, j)
                # print(cross)
                # print(vect1, vect2)
                # print(sub)
                # print(distances[i])
                # print(pos[inds])
                continue


            cross = cross/norm(cross)

            #check sign so not adding norm that's inward radially
            normRadDegDiffCheck = np.arccos(np.dot(cross, radDir))/np.pi*180.
            if normRadDegDiffCheck > 90:
            	cross *= -1

            norms[i] += cross
            totalCross += 1


        #average norm
        norms[i] /= totalCross

        #radDir = pos[i]/norm(pos[i])

        normRadDegDiff[i] = np.arccos(np.dot(norms[i], radDir))/np.pi*180.

        BDir = B[i]/norm(B[i])
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



R1_20050513 = readsav('20050513.R1')
R1_20050514 = readsav('20050514.R1')
R2_20050513 = readsav('20050513.R2')
R2_20050514 = readsav('20050514.R2')

R1_20050515 = readsav('20050515.R1')
R2_20050515 = readsav('20050515.R2')


arr_R1_20050513 = R1_20050513['arrayb']
arr_R1_20050514 = R1_20050514['arrayb']
arr_R2_20050513 = R2_20050513['arrayb']
arr_R2_20050514 = R2_20050514['arrayb']

arr_R1_20050515 = R1_20050515['arrayb']
arr_R2_20050515 = R2_20050515['arrayb']


allData = append(append(append(arr_R1_20050513[:,:-1], arr_R2_20050513[:,:-1], axis=0), append(arr_R1_20050514[:,:-1], arr_R2_20050514[:,:-1], axis=0), axis=1), append(arr_R1_20050515[:,:-1], arr_R2_20050515[:,:-1], axis=0), axis=1)




# allData = append(append(arr_R1_20050513[20:,:-1], arr_R2_20050513[:,:-1], axis=0), append(arr_R1_20050514[20:,:-1], arr_R2_20050514[:,:-1], axis=0), axis=1)
#allData = append(arr_R2_20050513[:,:-1], arr_R1_20050513[:,:-1], axis=0)

freqs1 = linspace(20., 1040., 256)
freqs2 = linspace(1075., 13825., 256)
freqs3 = linspace(14000, 25000, 23)
freqs = append(freqs1, freqs2)#when no interpolating, don't need higher things append(append(freqs1, freqs2), freqs3)

#starts at 100 khz
# freqs = freqs[20:]

print(shape(allData))
print(shape(freqs))
#
# days = ['20040106', '20040107', '20120102', '20120706', '20120119', '20120123']
# starts = array([6, 10, 14, 22, 14, 3])*60


#take up 16:00 on 5/13 to 00:00 5/16, so 48+8 hours
totMins = 60* 5 #5 #36 #5 #5 #32#56#60*32# +30 #5

# ind = 4
#
#
# day = days[ind]
#
#
# R1_day = readsav(day+'.R1')
# R2_day = readsav(day+'.R2')
#
#
# arr_R1_day = R1_day['arrayb']
# arr_R2_day = R2_day['arrayb']
#
# allData = append(arr_R1_day[20:,:-1], arr_R2_day[:,:-1], axis=0)
# #allData = append(arr_R2_20050513[:,:-1], arr_R1_20050513[:,:-1], axis=0)
#
# freqs1 = linspace(20., 1040., 256)
# freqs2 = linspace(1075., 13825., 256)
# freqs3 = linspace(14000, 25000, 23)
# freqs = append(append(freqs1, freqs2), freqs3)
#
# #starts at 100 khz
# freqs = freqs[20:]

newAllData = np.zeros((len(freqs), len(allData[0,:])))

#copy data for 13 MHZ for all the way up to 25
# newAllData[:-23, :] = allData
# for i in range(1,24):
#     newAllData[-1*i, :] = allData[-15, :]





newFreqs = linspace(100, 25000, 4096)

hours = linspace(0, 1440-1, 1440)/60.

hours = linspace(0, 2880-1, 2880)/60.

hours = linspace(0, 1440*3-1, 1440*3)/60.

X, Y = meshgrid(hours, freqs)

newX,newY = meshgrid(hours, newFreqs)

fact=1

#16:00 on 05/13/2005
start = 16*60 #+ 40 #starts[ind]

print(shape(X))
print(shape(Y))

X = X[:, start:start+totMins:fact]
Y = Y[:, start:start+totMins:fact]

print(shape(X))
print(shape(Y))


newX = newX[:, start:start+totMins:fact]
newY = newY[:, start:start+totMins:fact]
allData = allData[:, start:start+totMins:fact]
newAllData = newAllData[:, start:start+totMins:fact]

for i in range(len(newAllData[:,0])):
    for j in range(len(newAllData[0,:])):
        if newAllData[i][j] < 1.0 :
            newAllData[i][j] = 1.0

allDb = 10*log10(newAllData)




prePath = '../../'

# interpolates data from Wind frequencies to SunRISE frequencies
if not os.path.exists(prePath + 'newDbdata.txt'):
    newDb = my_interp(X, Y, allDb, newX, newY, spn=5)
    np.savetxt(prePath + 'newDbdata.txt', newDb)

else:
    newDb = np.loadtxt(prePath + 'newDbdata.txt')#+str(ind)+'.txt')

# print(shape(newDb))

#compute dB of sqrt(Intensity Ratio) after converting from dB to intensity again
newDbsqrt = 10*log10(np.sqrt(10.**(newDb/10)))

allDataDb = 10*log10(allData)

allDataDbsqrt = 10*log10(np.sqrt(10.**(allDataDb/10)))

print(shape(allDataDbsqrt))




vminp = -.9#-1.15
vmaxp = 20. #5#12. #5.#10.

sizein = 17

plt.rc('xtick',labelsize=sizein)
plt.rc('ytick',labelsize=sizein)

#take sqrt first then log

################################################33
#################freqfreqf


cpool = [ '#bd2309', '#bbb12d', '#1480fa', '#14fa2f', '#000000',
          '#faf214', '#2edfea', '#ea2ec4', '#ea2e40', '#cdcdcd',
          '#577a4d', '#2e46c0', '#f59422', '#219774', '#8086d9' ]
#black, red orange yellow Gren Blu purple white
colors = [(0, 0, 0),(1,0,0), (1, 127/255., 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (.5, 0, .5), (1,1,1)]
# (.5, 0, .5)
#black red yellow green green-blue blue purple white
colors = [(0, 0, 0),(1,0,0), (1, 1, 0) , (0, 1, 0), (0, 1, 1), (0, 0, 1), (0.5, 0, 0.5), (1,1,1)] #more colors, better contrast?
colors = [(0, 0, 0),(1,0,0), (1,0,0),(1, 1, 0), (1, 1, 0) ,(0, 1, 0), (0, 1, 0),(0, 1, 1), (0, 1, 1),(0, 0, 1),(0, 0, 1), (0.5, 0, 0.5),(0.5, 0, 0.5), (1,1,1)] #more colors, better contrast?


cm =LinearSegmentedColormap.from_list('my_list', colors, N=150)
# cm = colors.ListedColormap(cpool[0:15], 'indexed')



fig, ax = subplots(figsize=(16,8))

# p = ax.pcolormesh(newX[:-23, :], newY[:-23, :], newDbsqrt[:-23, :], vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
# p = ax.pcolormesh(X, Y, allDataDbsqrt, vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
p = ax.pcolormesh(X, Y, allDataDb, vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
ax.set_yscale("log")
ylabel('kHz', fontsize=sizein)
# xlabel('Hours past '+'2012/01/19'+' midnight', fontsize=sizein)
xlabel('Hours past '+'2005/05/13'+' midnight', fontsize=sizein)
title("Interpolated WIND Waves R1+R2 data", fontsize=sizein)
# xlim((newX.min(), newX.max()))
# ylim((min(freqs), max(freqs[:-23])))

xlim((X.min(), X.max()))
ylim((min(freqs), max(freqs)))

cb = fig.colorbar(p)

cb.set_label("dB of Intensity/Background", fontsize=sizein)



fig.savefig('allWindExtendSqrt'+'128short.png')#str(ind)+


# show()

# plt.close('all')

#returns ne cm^-3 in Leblanc, Dulk, and Bougeret model
def ldb(rs, aune = 7.2):
    nes = 2.8*10**5*rs**-2 + 3.5*10**6*rs**-4 + 6.8*10**8*rs**-6
    nes = nes*aune/7.2 #normalize
    return nes

#density model from reiner 2007 best fit
def reiner(rs, aune = 11.):
    AU=149597870700. #meters
    Rsun = 6.957E8 #meters
    aurs = AU/Rsun
    # aune = 11. #ne/cc
    nes = (rs/aurs)**-2 *aune
    # nes = 2.8*10**5*rs**-2 + 3.5*10**6*rs**-4 + 6.8*10**8*rs**-6
    return nes

#solar radius
rskm = 695510.

speed = 1000. #kms from manchester 2014 analysis, rough average


waitStart = 60 # for 16:47 #60 #how many minutes from start of data do we start the type ii


#not used here currently, later defined in loop setting
length = 31*60 #totMins - 60 #since we start 1 hour after #31*60 #totMins #30*60 # minutes
#now time to linearly deccelerate to 0 kms.  distinct from total time of data available to plot

#hours to match the 2d pcolormesh frame

totTypeMins = totMins - waitStart
typeStart = start + waitStart

xs = linspace(typeStart, typeStart + totTypeMins - 1, totTypeMins)/60.

# xs = linspace(17*60, 17*60 + totMins-1, totMins)/60.

startr = 7.0 #3.3 #2

# for speed in [4000, 2000, 1000, 750, 500, 250]:#, 75, 50, 25]:
#
#
#     speedm = speed*60
#     speedrsm = speedm/rskm
#
#     steprs = speedrsm
#
#
#     endr =  (length - 1)*steprs + startr
#     rs = linspace(startr, endr, length)
#     ldbnes = ldb(rs)
#
#
#     #neTokHz
#     khzs = 8.98*sqrt(ldbnes)
#
#     ax.plot(xs, khzs, label=str(speed)+' kms')
#
# legend()
# fig.savefig('allWindExtendSqrtConstSpeeds.png')#str(ind)+



speeds = speed*ones(totTypeMins)
rs = startr*ones(totTypeMins)

# speeds = speed*zeros(totMins)
# rs = startr*ones(totMins)

#speed starts at 150 km/s, linearly falls to 75 km/s after 15 hours where it stays, startr
speed2s = [(1000., 0., 31*60, 10.5), (750., 0., 31*60, 7.0), (500., 0., 31*60, 5.5), (1400., 0., 31*60, 6.1), (1200., 0., 31*60, 4.9), (1200., 0., 31*60, 3.7)]

#use reiner 2007 fit, where it uses fundamental and harmonic with coronagraph informed minimum speedDec

# $v_0 = 2500$ km/s, a deceleration of 26.7 m/s/s and a deceleration time of 17.9 hours, then steady 778 km/s until 1 AU
speed3s = [(2500., 778., 17*60 + 54, 3., 8.), (2500., 778., 17*60 + 54, 2.0, 11.), (2500., 778., 17*60 + 54, 1., 15.)]
# speed3s = [(2500., 778., 17*60 + 54, 2.0, 11.)]

khz = []

for i in range(len(speed3s)):


    speed1, speed2, length, startr, aune = speed3s[i]
    print(speed3s[i])

    speeds = speed2*ones(totTypeMins)
    rs = startr*ones(totTypeMins)
    # speeds = speed*zeros(totMins)
    # rs = startr*ones(totMins)

    speedDec = linspace(speed1, speed2, length + 1)
    # speedDec2 = linspace(speed1, 0, length + 1)

    # for j in range(length):
    #
    #     if j < trans*60:
    #         speeds[j] = speedDec[j]
    #
    #     else:
    #         speeds[j] = speed2


    # for j in range(length - 1):
    #     # rs[j+1] = rs[j] + speeds[j]*60/rskm
    #     rs[j+1] = rs[j] + speedDec2[j]*60/rskm
    #

    for j in range(totTypeMins -  1):
        if j < length:
            rs[j+1] = rs[j] + speedDec[j]*60/rskm
        else:
            rs[j+1] = rs[j] + speed2*60/rskm

    ldbnes = ldb(rs, 11.5) #use density model to go from distance to electron density per cc
    ldbnesr = reiner(rs, aune)
    khzs = 8.98*sqrt(ldbnes)
    khzsr = 8.98*sqrt(ldbnesr)
    khz.append(khzsr)
    decel = (1.*speed2-speed1)/(length/60.) #in km/hour acceleration/deceleration since its negative
    decelms = decel*1000./60./60. #decel in m/s
    # ax.plot(xs, khzs, label=str(startr) + ' Rs, ' + str(speed1)+' to ' + str(speed2) + ' km/s over ' + str(int(length)/60)  + ' hours')
    # ax.plot(xs, khzs, label='Fundamental r0 = ' + str(startr) + ' Rs, ' + 'v0 = ' + str(speed1)+' km/s, a =  %2.1f km/h'%decel)
    # ax.plot(xs, 2*khzs, label='Harmonic r0 = ' + str(startr) + ' Rs, ' + 'v0 = ' + str(speed1)+' km/s, a =  %2.1f km/h'%decel)

    # ax.plot(xs, khzs, label='Leblanc Fundamental r0 = ' + str(startr) + ' Rs, ')# + 'v0 = ' + str(speed1)+' km/s, a =  %2.1f m/s'%decelms + ' over %2.1f'%((length)/60.) + ' hours')
    # ax.plot(xs, 2*khzs, 'm--', label='Leblanc Harmonic r0 = ' + str(startr) + ' Rs, ' )#+ 'v0 = ' + str(speed1)+' km/s, a =  %2.1f m/s'%decelms + ' over %2.1f'%((length)/60.) + ' hours')

    if i == 0:
        ax.plot(xs, khzsr, 'g-', label= r'Fundamental, $r_0$ = ' + str(startr) + r' $R_S$, ' + r'$n_e$/cc @ 1 AU = ' + str(aune))# r'$v_0$ = ' + str(speed1)+' km/s, a =  %2.1f m/s'%decelms + ' over %2.1f'%((length)/60.) + ' hours')
    elif i == 2:
        ax.plot(xs, khzsr, 'm-', label= r'Fundamental, $r_0$ = ' + str(startr) + r' $R_S$, ' + r'$n_e$/cc @ 1 AU = ' + str(aune))# r'$v_0$ = ' + str(speed1)+' km/s, a =  %2.1f m/s'%decelms + ' over %2.1f'%((length)/60.) + ' hours')
    else:
        ax.plot(xs, khzsr, 'b-', label= r'Fundamental, $r_0$ = ' + str(startr) + r' $R_S$, ' + r'$n_e$/cc @ 1 AU = ' + str(aune))# r'$v_0$ = ' + str(speed1)+' km/s, a =  %2.1f m/s'%decelms + ' over %2.1f'%((length)/60.) + ' hours')

    # ax.plot(xs, khzsr, 'b-', label= r'Fundamental, $r_0$ = ' + str(startr) + r' $R_S$, ' + r'$v_0$ = ' + str(speed1)+' km/s, a =  %2.1f m/s'%decelms + ' over %2.1f'%((length)/60.) + ' hours')
    # ax.plot(xs, 2*khzsr, 'b--', label= r'Harmonic,       $r_0$ = ' + str(startr) + r' $R_S$, ' + r'$v_0$ = ' + str(speed1)+' km/s, a =  %2.1f m/s'%decelms + ' over %2.1f'%((length)/60.) + ' hours')

ax.legend()
fig.savefig('windSpeedProfiles.png')


# show()

# fig.savefig('allWindExtendSqrtSpeedto0long.png')#str(ind)+




#new code

#fast lower frequency, middle, slower higher frequency, then second set good for close to the beginning
khz1, khz2, khz3 = khz

#after index 138, khz1 > khz4
#after index  54, khz2 > khz5
#after index 30, khz3 >  khz6

# ind1 = where(khz4>khz1)[0][-1]
# ind2 = where(khz5>khz2)[0][-1]
# ind3 = where(khz6>khz3)[0][-1]

lows = khz1.copy()
mids = khz2.copy()
highs = khz3.copy()


# for i in range(ind1+1):
#     lows[i] = khz4[i]
#
# for i in range(ind2+1):
#     mids[i] = khz5[i]
#
# for i in range(ind3+1):
#     highs[i] = khz6[i]


lowsF = lows.copy()
highsF = highs.copy()
midsF = mids.copy()


fig, ax = subplots(figsize=(16,8))

# p = ax.pcolormesh(newX[:-23, :], newY[:-23, :], newDbsqrt[:-23, :], vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
# p = ax.pcolormesh(X, Y, allDataDbsqrt, vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
p = ax.pcolormesh(X, Y, allDataDb, vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
ax.set_yscale("log")
ylabel('kHz', fontsize=sizein)
# xlabel('Hours past '+'2012/01/19'+' midnight', fontsize=sizein)
xlabel('Hours past '+'2005/05/13'+' midnight', fontsize=sizein)
title("Interpolated WIND Waves R1+R2 data", fontsize=sizein)
# xlim((newX.min(), newX.max()))
# ylim((min(freqs), max(freqs[:-23])))

xlim((X.min(), X.max()))
ylim((min(freqs), max(freqs)))

cb = fig.colorbar(p)

cb.set_label("dB of Intensity/Background", fontsize=sizein)


ax.plot(xs, lows, 'w--')
ax.plot(xs, mids, 'w-')
ax.plot(xs, highs, 'w--')

fig.savefig('allWindExtendSqrtSpeedto0longoutline.png')#str(ind)+



stencil = zeros(shape(allDataDbsqrt))
stencilNorm = zeros(shape(allDataDbsqrt))

for i in range(totTypeMins):
    ii = i + waitStart

    midFreqInd = bisect.bisect_right(freqs, mids[i])
    midFreq = freqs[midFreqInd] #kHz

    lowFreqInd = bisect.bisect_right(freqs, lows[i])
    lowFreq = freqs[lowFreqInd] #kHz

    highFreqInd = bisect.bisect_right(freqs, highs[i])
    highFreq = freqs[highFreqInd] #kHz


    sig3up = highFreqInd - midFreqInd
    sig3down = midFreqInd - lowFreqInd

    sigup = sig3up/3.
    sigdown = sig3down/3.

    stencil[midFreqInd, ii] = max(normVar.pdf(midFreqInd, loc=midFreqInd, scale=sigdown), normVar.pdf(midFreqInd, loc=midFreqInd, scale=sigup))
    stencilNorm[midFreqInd, ii] = 1.0 #norm.pdf(midFreqInd., loc=midFreqInd, scale=sigdown) #1.0

    for j in range(lowFreqInd, midFreqInd):
        stencil[j, ii] = normVar.pdf(j, loc=midFreqInd, scale=sigdown)
        stencilNorm[j, ii] = normVar.pdf(j, loc=midFreqInd, scale=sigdown)/ normVar.pdf(midFreqInd, loc=midFreqInd, scale=sigdown) #stencil[midFreqInd, ii]

        stencil[j, ii] = normVar.pdf(freqs[j], loc=freqs[midFreqInd], scale=(midFreq - lowFreq)/3.)
        stencilNorm[j, ii] = normVar.pdf(freqs[j], loc=freqs[midFreqInd], scale=(midFreq - lowFreq)/3.)/ normVar.pdf(freqs[midFreqInd], loc=freqs[midFreqInd], scale=(midFreq - lowFreq)/3.) #stencil[midFreqInd, ii]


    for j in range(midFreqInd+1, highFreqInd+1):
        stencil[j, ii] = normVar.pdf(j, loc=midFreqInd, scale=sigup)
        stencilNorm[j, ii] = normVar.pdf(j, loc=midFreqInd, scale=sigup)/ normVar.pdf(midFreqInd, loc=midFreqInd, scale=sigup)

        stencil[j, ii] = normVar.pdf(freqs[j], loc=freqs[midFreqInd], scale=(highFreq - midFreq)/3.)
        stencilNorm[j, ii] = normVar.pdf(freqs[j], loc=freqs[midFreqInd], scale=(highFreq - midFreq)/3.)/ normVar.pdf(freqs[midFreqInd], loc=freqs[midFreqInd], scale=(highFreq - midFreq)/3.) #stencil[midFreqInd, ii]

    if i == 10 or i == 20 or i == 30:
        #make plot of gaussian profile over spectral Intensity
        fig, ax = subplots(figsize=(16,8))
        ax.set_xlabel('Frequency (kHz)', fontsize=sizein)
        ax.plot(Y[:,ii], allDataDb[:,ii], 'r-')
        ax.set_ylabel('Wind Spectral Intensity dB over Background', color = 'r', fontsize=sizein)
        ax.set_ylim((0, 1.5)) #amax(allDataDb[:,ii])))
        ax.tick_params(axis ='y', labelcolor = 'r')
        # ax.set_xscale('log')
        ax2 = ax.twinx()
        ax2.plot(Y[:,ii], stencilNorm[:,ii], 'b-')#, label='t=0 Double Sided Gaussian Stencil') # freqs on X axis
        ax2.set_ylabel('Gaussian Profile', color = 'b', fontsize=sizein)
        ax2.set_ylim((0, amax(stencilNorm[:,ii])))
        ax2.tick_params(axis ='y', labelcolor = 'b')

        title("Single t=%d Gaussian Profile vs Wind/WAVES Intensity"%i, fontsize=sizein+4)
        fig.savefig('singleStencilSpectra%d.png'%i)
        clf()



fig, ax = subplots(figsize=(16,8))

# p = ax.pcolormesh(newX[:-23, :], newY[:-23, :], newDbsqrt[:-23, :], vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
p = ax.pcolormesh(X, Y, stencilNorm, vmin=0., vmax=amax(stencilNorm),cmap=cm)#gist_ncar_r")
ax.set_yscale("log")
ylabel('kHz', fontsize=sizein)
# xlabel('Hours past '+'2012/01/19'+' midnight', fontsize=sizein)
xlabel('Hours past '+'2005/05/13'+' midnight', fontsize=sizein)
title("Type II Gaussian Stencil, Column Normalized", fontsize=sizein)
# xlim((newX.min(), newX.max()))
# ylim((min(freqs), max(freqs[:-23])))

xlim((X.min(), X.max()))
ylim((min(freqs), max(freqs)))

cb = fig.colorbar(p)

cb.set_label("Column Normalized Weight (Gaussian)", fontsize=sizein)


ax.plot(xs, lows, 'w--')
# ax.plot(xs, mids, 'w.')
ax.plot(xs, highs, 'w--')

fig.savefig('stencilLongNorm.png')#str(ind)+



fig, ax = subplots(figsize=(16,8))

# p = ax.pcolormesh(newX[:-23, :], newY[:-23, :], newDbsqrt[:-23, :], vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
p = ax.pcolormesh(X, Y, stencil, vmin=0., vmax=amax(stencil),cmap=cm)#gist_ncar_r")
ax.set_yscale("log")
ylabel('kHz', fontsize=sizein)
# xlabel('Hours past '+'2012/01/19'+' midnight', fontsize=sizein)
xlabel('Hours past '+'2005/05/13'+' midnight', fontsize=sizein)
title("Type II Gaussian Stencil, Gaussian Weights", fontsize=sizein)
# xlim((newX.min(), newX.max()))
# ylim((min(freqs), max(freqs[:-23])))

xlim((X.min(), X.max()))
ylim((min(freqs), max(freqs)))

cb = fig.colorbar(p)

cb.set_label("Weight (Gaussian)", fontsize=sizein)


ax.plot(xs, lows, 'w--')
# ax.plot(xs, mids, 'w.')
ax.plot(xs, highs, 'w--')

# fig.savefig('stencilLong.png')#str(ind)+


#bring in interpolated cme2dhist data interpolated to the cadence of Wind

#now make gaussian for each time from model lines
#count distance in frequency channels between lines at each time, make that 2 sigma.

#create function to create 1 D index of how well it overlaps with truth DATA

#create loops for each square degree data cut, make list of 10 top scorers

#figures of 1 true wind data 2 idealized gaussian 3 synthetic spectra 4 highlighted list and score,


##copying from cme2dhist code colatitude


freqs0 = zeros(len(freqs)+1)
freqs0[1:] = freqs

lows = freqs0[:-1]
highs = freqs0[1:]

modes = ['Front', 'Flank', 'Sheet', 'Lobe'] #FrontBit and FlankBit also defined as modes, that slice up in a defined way via an index

datPrefix = ['IsoEntropy=4_', 'isoNe=3.5_', 'iso_Tp=3.5MK_', 'isoNe=3.5_']


# (x, y, z, ux, uy, uz, Bx, By, Bz, f, n1, n2, n3, fv1, fv2, fv3, ft, tbn, Nx, Ny, Nz, NDiff, p (pressure), npsingle (electron density), rho (density), ent, entrat, fluxf, fluxh)

    #thetabn cut,17
cuts = []
# cuts = [(17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'Front', 0), (17, 0, 90, 'Shock Flank', 'isoNe=3.5_', 'Flank', 0), (17, 0, 90, 'Current Sheet', 'iso_Tp=3.5MK_', 'Sheet', 0), (17, 0, 90, 'Diffuse Lobe', 'isoNe=3.5_', 'Lobe', 0), \
#         (17, 60, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'FrontHighTheta', 0)]
#
# cuts.append((17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'FrontBit', 0))
# cuts.append((17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'Front', 0))
#
# cuts.append((17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'FrontBit', 662))
# cuts.append((17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'FrontBit', 663))
# cuts.append((17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'FrontBit', 690))
# cuts.append((17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'FrontBit', 732))
# cuts.append((17, 50, 90, r'Front of Shock, $\Theta_{Bn} > 50^{\circ}$', 'IsoEntropy=4_', 'FrontBit', 662))
# cuts.append((17, 50, 90, r'Front of Shock, $\Theta_{Bn} > 50^{\circ}$', 'IsoEntropy=4_', 'FrontBit', 663))

# cuts.append((16, 2250., 100000, 'de Hoffmann-Teller Velocity 2250 km/s +', 'IsoEntropy=4_', 'FrontBit', 663))

# cuts.append((16, 2500., 9999999., 'de Hoffmann-Teller Velocity 2500 km/s +', 'IsoEntropy=4_', 'Front', 25))
# cuts.append((16, 2000., 100000, 'de Hoffmann-Teller Velocity 2000 km/s +', 'IsoEntropy=4_', 'FrontBit', 810))

#1600 bits, 3x3 degrees from -60 to 60
# linspace(13*40, 20*40, 60)  #gets a range of good and bad bits for inspection, range(1, 1601) gets all
# for i in linspace(13*40+2, 20*40+2, 60): #range(1, 1601): #linspace(13*40, 20*40, 60):#[0]:#[825, 826]: #[820, 824, 816, 700, 704, 696, 940, 944, 936]:#range(1, 1601):#5, 10, 15]:#range(0, 1):
#     cuts.append((17, 0, 90, 'Entropy Shock Front', 'IsoEntropy=4_', 'FrontBit', int(i)))
#
#
# cuts.append((17, 0, 90, 'Density Shocked Region', 'isoNe=3.5_', 'FlankBit', 0))
#
# cuts.append((17, 0, 90, 'Current Sheet Region', 'iso_Tp=3.5MK_', 'Sheet', 0))


#check the highest thetaBN
# cuts.append((17, 0, 90, 'High Theta_Bn', 'IsoEntropy=4_', 'Front', 60))
# cuts.append((17, 70, 90, r'Front of Shock, $\Theta_{Bn} > 70^{\circ}$', 'IsoEntropy=4_', 'Front', 70))
# cuts.append((17, 80, 90, 'Highest Theta_Bn', 'IsoEntropy=4_', 'Front', 80))

# cuts.append((17, 45, 90, r'Front of Shock, $\Theta_{Bn} > 45^{\circ}$', 'IsoEntropy=4_', 'Front', 45))

# cuts.append((17, 50, 90, r'Front of Shock, $\Theta_{Bn} > 50^{\circ}$', 'IsoEntropy=4_', 'Front', 50))
# cuts.append((17, 60, 90, r'Front of Shock, $\Theta_{Bn} > 60^{\circ}$', 'IsoEntropy=4_', 'Front', 60))
# cuts.append((17, 70, 90, r'Front of Shock, $\Theta_{Bn} > 70^{\circ}$', 'IsoEntropy=4_', 'Front', 70))
# cuts.append((17, 80, 90, r'Front of Shock, $\Theta_{Bn} > 80^{\circ}$', 'IsoEntropy=4_', 'Front', 80))
# cuts.append((16, 3000., 100000, 'de Hoffmann-Teller Velocity 3000 km/s +', 'IsoEntropy=4_', 'Front', 3))
# cuts.append((16, 4000., 100000, 'de Hoffmann-Teller Velocity 4000 km/s +', 'IsoEntropy=4_', 'Front', 4))
# cuts.append((16, 5000., 100000, 'de Hoffmann-Teller Velocity 5000 km/s +', 'IsoEntropy=4_', 'Front', 5))

# cuts.append((17, 60, 90, 'High Theta_Bn', 'isoNe=3.5_', 'Flank', 60))
# cuts.append((17, 70, 90, 'Higher Theta_Bn', 'isoNe=3.5_', 'Flank', 70))
# cuts.append((17, 80, 90, 'Highest Theta_Bn', 'isoNe=3.5_', 'Flank', 80))


cuts.append((17, 0, 90, 'All Data', '2005CME_iso_Machms=3_', 'Flank', 0))
cuts.append((17, 60, 90, 'Higher Theta_Bn', '2005CME_iso_Machms=3_', 'Flank', 60))
cuts.append((17, 80, 90, 'Highest Theta_Bn', '2005CME_iso_Machms=3_', 'Flank', 80))


txtPrePath = "/mnt/LinuxData/CME_Data/"

maxScores = []

zeroScores = []

for cut in cuts:

    varInd, varLow, varHigh, label, datPre, mode, bitInd = cut

    #bitInd for indexing through the bits of the other modes.  Make 0 uncut

    print(cut)

    if datPre == 'iso_Tp=3.5MK_':
        times1 = linspace(2, 36, 18).astype(int) #every 2
        times2 = linspace(40, 80, 11).astype(int) #every 4
        times3 = np.array([85, 90, 92, 95, 100, 105, 110, 115, 120])
        times = np.append(np.append(times1, times2), times3)

    else:
        times1 = linspace(8, 36, 15).astype(int) #every 2
        times2 = linspace(40, 80, 11).astype(int) #every 4
        times3 = np.array([85, 90, 92, 95, 100, 105, 110, 115, 120])
        newMins = array([ 180,  480,  600,  720,  840,  960, 1080, 1200, 1320, 1440, 1560, 1680, 1800, 2160])
        times = np.append(np.append(times1, times2), times3)
        # times = np.append(np.append(times1, times2), np.append(times3, newMins))



    plt.close('all')
    # mins = linspace(0, len(times)-1, len(times))

    Xsim, Ysim = meshgrid(times, freqs)
    #ax1.set_title('2D Histogram of Simulated Radio Activity over Time and Frequency \n Data Cut of '+label)#, y = 1.2)

    cutData = np.zeros((len(times), len(freqs)))
    cutDataTot = np.zeros((len(times), len(freqs)))
    cutDataCol1 = np.zeros((len(times), len(freqs)))

    # cutData = np.zeros((totMins, len(lows)))
    # cutDataTot = np.zeros((totMins, len(lows)))

    maxrs = []
    meanrs = []
    meanxs = []
    meanys = []
    meanzs = []
    medianrs = []

    if datPre[:4] == '2005':
        times = np.array([10, 20, 30, 40, 60, 90]) #only use for limited Mach data
        Xsim, Ysim = meshgrid(times, freqs)
        #ax1.set_title('2D Histogram of Simulated Radio Activity over Time and Frequency \n Data Cut of '+label)#, y = 1.2)

        cutData = np.zeros((len(times), len(freqs)))
        cutDataTot = np.zeros((len(times), len(freqs)))
        cutDataCol1 = np.zeros((len(times), len(freqs)))

    for t in range(len(times)):

        plt.close('all')

        time = times[t]


        datFile = datPre + str(time) +'min.dat'

        if datFile[:4] != '2005':
            if time < 10:
                datFile = datPre + '00' + str(time) +'min.dat'
            elif time < 100:
                datFile = datPre + '0' + str(time) +'min.dat'

        if datPre == 'iso_Tp=3.5MK_':
            data =  json.load(open(txtPrePath+datFile+"data.txt", 'r'))
        else:
            data =  json.load(open(txtPrePath+datFile+"dataTBN.txt", 'r'))

        if time == 180:
            dataIH =  json.load(open(txtPrePath+"IsoEntropy=4_180minIH.datdata.txt", 'r'))
            dataSC =  json.load(open(txtPrePath+"IsoEntropy=4_180minSC.datdata.txt", 'r'))
            data = np.append(dataIH, dataSC, axis=0)
            data = data.tolist()

            data =  json.load(open(txtPrePath+"IsoEntropy=4_180minSC.datdata.txt", 'r'))

        #sorts by x dir & convert to degrees from Rs
        data=np.array(sorted(data))

        # N = np.shape(data)[0]
        # Ns = data[:, 10:13]
        # lovelos = data[:, 3:6]
        # lomagField = data[:, 6:9]
        #
        # HTVelo = np.zeros([N, 3])
        # # fluxF = np.zeros(N)
        # # fluxH = np.zeros(N)
        # for i in range(N):
        #     vup = lovelos[i] - Ns[i]*1500.
        #     vnif = np.cross(Ns[i], np.cross(vup,Ns[i]))
        #     bnif = np.cross(Ns[i], np.cross(lomagField[i],Ns[i]))
        #     numer = np.cross(Ns[i], np.cross(vnif, bnif))
        #     numer = np.cross(Ns[i], np.cross(vup, lomagField[i]))
        #     denom = np.dot(lomagField[i], Ns[i])
        #     # fluxF[i], fluxH[i] = getflux_Cairns(Tis[i], Tes[i], thetaBN[i], velos[i] - lovelos[i], lomagField[i], magField[i])
        #     HTVelo[i] = numer/denom
        # #
        # absHTVelo = np.apply_along_axis(linalg.norm, 1, HTVelo)#norm(HTVelo, axis=1)
        #
        # data[:, 13:16] = HTVelo
        # data[:, 16] = absHTVelo
        print('time is ' +  str(time))

        print(str(np.shape(data)[0]) + ' is numpoints after loading time ' + str(time))

        deg2D = data[:, 1:3]
        visData = data.copy()

        #additional hi mag cut, artifact?

        #do cut from 4panel
        if mode == 'Front':

            bmag = np.sqrt(visData[:, 6]**2 + visData[:, 7]**2 + visData[:, 8]**2)
            # bmag = np.sqrt(tempData[:, 6]**2 + tempData[:, 7]**2 + tempData[:, 8]**2)
            maxmag = max(bmag)
            bigmag = np.where(bmag > 0.1*maxmag)
            visData = np.delete(visData, bigmag, axis=0)


            # if len(visData) != 0:
            #     r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
            #     tinymax = max(r3s)
            #     smallData = np.where(r3s < 0.7*tinymax)
            #     visData = np.delete(visData, smallData, axis=0)


            ###alternate version, using better cut for artifact generally used in frontbit

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
            longs = np.arctan2(visData[:, 1], visData[:, 0])/np.pi*180.
            lats = np.arcsin(visData[:, 2]/ r3s)/np.pi*180.

            #remove close in artifact
            maxr = max(r3s)
            print('max r out projected is ' + str(maxr))
            print('mean r is ' + str(np.mean(r3s)))

            #convoluted removal?
            longLows = linspace(-60., 57., 40)
            longHighs = longLows + 3.0

            latLows = linspace(-60., 57., 40)
            latHighs = latLows + 3.0

            # longLows = linspace(-80., 76., 40)
            # longHighs = longLows + 4.
            #
            # latLows = linspace(-80., 76., 40)
            # latHighs = latLows + 4.

            smallR = np.array([])
            tinyRs = zeros(1600)

            #cycle through bit by bit to remove stuff under the maximum of that solar sky
            # for itInd in range(1600):
            #     badAzi = np.where(np.logical_or(np.logical_or(lats < latLows[itInd/40], lats > latHighs[itInd/40]), np.logical_or(longs < longLows[itInd%40], longs > longHighs[itInd%40])))[0]
            #     tinyArea = np.delete(visData, badAzi, axis=0)
            #     if len(tinyArea) == 0:
            #         continue
            #     r3stiny = sqrt(tinyArea[:, 0]**2 + tinyArea[:, 1]**2 + tinyArea[:, 2]**2)
            #     maxrtiny = max(r3stiny)
            #     tinyRs[itInd] = maxrtiny
            #
            # for itInd in range(1600):
            #     if tinyRs[itInd] == 0.:
            #         continue
            #     #use contrapositive to pick out points from only 1 long bin, and go over relative max
            #     sR = np.where(np.logical_and(r3s < 0.9*tinyRs[itInd], np.logical_and(np.logical_and(lats > latLows[itInd/40], lats < latHighs[itInd/40]), np.logical_and(longs > longLows[itInd%40], longs < longHighs[itInd%40]))))[0]
            #
            #     smallR = np.append(smallR, sR)

            #delete secondary population behind main shock, as well as artifacts near sun
            # smallR = np.where(np.logical_or(r2s < 2., np.logical_and(r2s < maxr*.7, np.logical_and(azimuths > 20., azimuths < 70.))))[0]
            # smallR = np.where(np.logical_or(r3s < 2., np.logical_or(r3s < maxr*.8, np.logical_and(lats > 20., lats < 70.))))[0]

            visData = np.delete(visData, smallR, axis=0)

            #refresh derived data after cut
            deg2D = visData[:, 1:3]
            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
            longs = np.arctan2(visData[:, 1], visData[:, 0])/np.pi*180.
            lats = np.arcsin(visData[:, 2]/ r3s)/np.pi*180.

            #do 0-20 azimuth part of shock
            # largeLong = np.where(np.logical_or(longs < -10., longs > 0.))[0]

            # largeLong = np.where(r3s < 2.0 )[0]

            # visData = np.delete(visData, largeLong, axis=0)

            data = visData.copy()


        #do cut from 4panel
        #
        if mode == 'FrontBit':

            bmag = np.sqrt(visData[:, 6]**2 + visData[:, 7]**2 + visData[:, 8]**2)
            # bmag = np.sqrt(tempData[:, 6]**2 + tempData[:, 7]**2 + tempData[:, 8]**2)
            maxmag = max(bmag)
            bigmag = np.where(bmag > 0.1*maxmag)
            visData = np.delete(visData, bigmag, axis=0)


            if len(visData) != 0:
                r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
                tinymax = max(r3s)
                smallData = np.where(r3s < 0.7*tinymax)
                visData = np.delete(visData, smallData, axis=0)


            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)

            longs = np.arctan2(visData[:, 1], visData[:, 0])/np.pi*180.
            lats = np.arcsin(visData[:, 2]/ r3s)/np.pi*180.

            #remove close in artifact
            maxr = max(r3s)
            print('max r out is ' + str(maxr))

            maxrs.append(maxr)
            meanrs.append(np.mean(r3s))
            meanxs.append(np.mean(visData[:, 0]))
            meanys.append(np.mean(visData[:, 1]))
            meanzs.append(np.mean(visData[:, 2]))
            medianrs.append(np.median(r3s))
            # print('mean r out is ' + str(np.mean(r3s)))
            # print('median r out is ' + str(np.median(r3s)))

            #delete secondary population behind main shock, as well as artifacts near sun
            # smallR = np.where(np.logical_or(r2s < 2., np.logical_and(r2s < maxr*.7, np.logical_and(azimuths > 20., azimuths < 70.))))[0]
            # smallR = np.where(np.logical_or(r2s < 2., np.logical_and(r2s < maxr*.7, np.logical_and(azimuths > 20., azimuths < 70.))))[0]

            # if len(visData) != 0:
            #     r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
            #     tinymax = max(r3s)
            #     smallData = np.where(r3s < 0.8*tinymax)
            #     visData = np.delete(visData, smallData, axis=0)


            # visData = np.delete(visData, smallR, axis=0)

            #refresh derived data after cut
            # deg2D = visData[:, 1:3]
            # azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            # r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)



            #do uncut
            if bitInd == 0:
                label = 'Shock Front Bit Ind ' + str(bitInd) + ' \nFull Clean Cut '

                # if len(visData) != 0:
                #     r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
                #
                #     tinymax = max(r3s)
                #
                #     smallData = np.where(r3s < 0.8*tinymax)
                #     visData = np.delete(visData, smallData, axis=0)

                data = visData.copy()



            else:

                #only uncut and 1-100 others allowed
                #3x3 degree bins of long lat.  -60+60 degree range 40x40 bins, 1600 total
                maxAllowed = 1600
                bitInd = min(bitInd, maxAllowed)
                bitInd = bitInd - 1

                longLows = linspace(-60., 57., 40)
                longHighs = longLows + 3.0

                latLows = linspace(-60., 57., 40)
                latHighs = latLows + 3.0

                #arranged so lines of constant latitude are done in succession
                badAzi = np.where(np.logical_or(np.logical_or(lats < latLows[bitInd/40], lats > latHighs[bitInd/40]), np.logical_or(longs < longLows[bitInd%40], longs > longHighs[bitInd%40])))[0]

                visData = np.delete(visData, badAzi, axis=0)

                #this if cut makes the difference between zeroscores 1 (commmented) and zeroscores2 (uncommented) zeroscores 2 is better, smoother heatmap
                # if len(visData) != 0:
                #     r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
                #
                #     tinymax = max(r3s)
                #
                #     smallData = np.where(r3s < 0.8*tinymax)
                #     visData = np.delete(visData, smallData, axis=0)

                data = visData.copy()

                strbit = '\n Longitude %2.0f - %2.0f degrees, Latitude %2.0f - %2.0f'%(longLows[bitInd%40], longHighs[bitInd%40], latLows[bitInd/40], latHighs[bitInd/40])

                bitInd = bitInd+1

                label = 'Entropy Shock Flank Bit Ind ' + str(bitInd) + strbit




        if mode == 'Flank':

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)

            maxr = max(r2s)


            smallAzi = np.where(azimuths > -30.)[0]

            # visData = np.delete(visData, smallAzi, axis=0)

            deg2D = visData[:, 1:3]

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            if len(r2s) == 0:
                continue


            maxr = max(r2s)
            print('max r out projected is ' + str(maxr))
            smallR = np.where(r2s < maxr/2.)[0]

            # visData = np.delete(visData, smallR, axis=0)
            #
            # data = visData.copy


        #do cut from 4panel
        if mode == 'FlankBit':

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)

            longs = np.arctan2(visData[:, 1], visData[:, 0])/np.pi*180.
            lats = np.arcsin(visData[:, 2]/ r3s)/np.pi*180.

            #remove close in artifact
            maxr = max(r2s)
            print('max r out projected is ' + str(maxr))

            #delete secondary population behind main shock, as well as artifacts near sun
            # smallR = np.where(np.logical_or(r2s < 2., np.logical_and(r2s < maxr*.7, np.logical_and(azimuths > 20., azimuths < 70.))))[0]
            # smallR = np.where(np.logical_or(r2s < 2., np.logical_and(r2s < maxr*.7, np.logical_and(azimuths > 20., azimuths < 70.))))[0]


            # visData = np.delete(visData, smallR, axis=0)
            #
            # #refresh derived data after cut
            # deg2D = visData[:, 1:3]
            # azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            # r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)



            #do uncut
            if bitInd == 0:
                label = 'Shock Flank Bit Ind ' + str(bitInd) + ' \nFull Density Enhancement Uncut '
                data = visData.copy()



            else:

                #only uncut and 1-100 others allowed
                maxAllowed = 200
                bitInd = min(bitInd, maxAllowed)
                bitInd = bitInd - 1


                #define 1 degree chunks between -40 -39 to positive 59-60, 100 total
                aziLows = linspace(-40., 55., maxAllowed/10)
                aziHighs = aziLows + 5.0

                rLows = linspace(0., 0.9, maxAllowed/20)
                rHighs = rLows + 0.1



                badAzi = np.where(np.logical_or(np.logical_or(r2s < maxr*rLows[bitInd/20], r2s > maxr*rHighs[bitInd/20]), np.logical_or(azimuths < aziLows[bitInd%20], azimuths > aziHighs[bitInd%20])))[0]

                visData = np.delete(visData, badAzi, axis=0)

                data = visData.copy()

                strbit = '\n Azimuth range %2.2f - %2.2f degrees, r percentile %2.0f - %2.0f'%(aziLows[bitInd%20], aziHighs[bitInd%20], 100*rLows[bitInd/20], 100*rHighs[bitInd/20])

                bitInd = bitInd+1

                label = 'Density Shock Flank Bit Ind ' + str(bitInd) + strbit







        if mode == 'Sheet':

            ALLrs = np.sqrt(visData[:,0]**2 + visData[:,1]**2 + visData[:,2]**2)
            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)

            longs = np.arctan2(visData[:, 1], visData[:, 0])/np.pi*180.
            lats = np.arcsin(visData[:, 2]/ r3s)/np.pi*180.


            #scale backnose of entropy shock data back towards the sun, adjusting electron denisty and plas freq
            if time > 1:
                bigX = np.where(ALLrs > 3.5)[0]
                visData = np.delete(visData, bigX, axis=0)

            # if time >= 65: #clean bad data point where only 1 freqpoint throws off colorscale
            #     bigX = np.where(ALLrs > 0.01)[0]
            #     visData = np.delete(visData, bigX, axis=0)

            data = visData.copy()



        if mode == 'Lobe':
            # use isoNe files,

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            maxr = max(r2s)
            print('max r out projected is ' + str(maxr))
            smallR = np.where(np.logical_or(r2s < maxr*1./4., r2s > maxr*3./4.))[0]

            visData = np.delete(visData, smallR, axis=0)

            data = visData.copy()

            ######one cut cut
            #done cut

        print(str(np.shape(data)[0]) + ' is numpoints after cuts')
        # print('time is ' +  str(time))





        #fix thetaBN to 0-90 instead of 0-180
        for i in range(np.shape(data)[0]):
            if data[i, 17] > 90.:
                data[i, 17] = 180. - data[i, 17]


        freqData = data[:, 9]*1000 # *0.5 #flag for possible upstream factor
        varIndData = data[:, varInd]

        # print('len of freqdata is ' + str(len(freqData)))
        # BData = np.sqrt(data[:, 6]**2 + data[:, 7]**2 + data[:, 8]**2)
        # Bmax = 0.
        # if len(data) != 0:
        #     Bmax = max(BData)
        #
        # varIndData = BData
        # varLow = Bmax*0.7
        # varHigh = Bmax


        # print('sorting data by x coord')
        #data = [(x, y, z, f) for x, y, z, f in zip(xs, ys, zs, fs)]

        #project to 2D y-z POS
        deg2D = data[:, 1:3]

        N = len(data[:, 0])


        pttot = 0
        ptmax = 0

        allGoodInds = np.array([], dtype=np.integer)

        for f in range(len(freqs)):
            freqlo = lows[f]
            freqhi = highs[f]
            # if freqhi == highs[-1]:
            #     freqhi = 100.

            df = freqhi - freqlo
            #freqRangeInds = np.where(np.logical_and(fs > freqlo, fs < freqhi))[0]

            #freqRangeInds = np.where(np.logical_and(varColor >= varLow, varColor <= varHigh))[0]
            freqOnlyRangeInds = np.where(np.logical_and(freqData > freqlo, freqData < freqhi))
            freqRangeInds = np.where(np.logical_and(np.logical_and(freqData > freqlo, freqData < freqhi), np.logical_and(varIndData >= varLow, varIndData <= varHigh)))[0]

            allGoodInds = np.append(allGoodInds, freqRangeInds)

            cutData[t, f] = len(freqRangeInds)/float(df)
            cutDataTot[t, f] = len(freqRangeInds)
            cutDataCol1[t, f] = len(freqRangeInds)

            pttot  = pttot + len(freqRangeInds)

            if ptmax < cutData[t, f]:
                ptmax = cutData[t, f]
        if ptmax != 0.:
            cutData[t, :] /= float(ptmax) #float(pttot)
        print('pttotal after chosen variable cut is ' + str(pttot))
        #normalize by numbner of points in file

        #normalize so sum of whole column is 1, and it shows proportionally the freq spread in the cut
        if pttot !=0:
            cutDataCol1[t, :] /= float(pttot)

        # if this is true, a 2D and 3D movie of the cut is made, pngs and gif
        makeSnaps = True


        if makeSnaps:
            #make 3d picture, make gif of cut truth at the end
            figg = plt.figure()
            axg = figg.add_subplot(111, projection='3d')

            #add spherical sun
            # draw sphere
            su, sv = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            sx = np.cos(su)*np.sin(sv)
            sy = np.sin(su)*np.sin(sv)
            sz = np.cos(sv)
            axg.plot_surface(sx, sy, sz, color="y")

            #[freqRangeInds]

            skip = 1

            cmapg = plt.get_cmap('jet', 512)
            cmaplistg = [cmapg(i) for i in range(cmapg.N)]

            # freqColors = []
            # for j in range(np.shape(data)[0]):
            #      li = bisect.bisect_left(freqs, visData[j,9]*1000.)
            #
            #      freqColors.append(cmapg(li))


            numPts = len(allGoodInds)

            freqColors = []
            for j in range(numPts):
                 li = bisect.bisect_left(freqs, visData[:,9][allGoodInds[j]]*1000.)

                 freqColors.append(cmapg(li))

            p=axg.scatter(visData[:,0][allGoodInds], visData[:,1][allGoodInds], visData[:,2][allGoodInds], color=freqColors)

            # axg.set_xlim(-10, 200)
            # axg.set_ylim(-150, 150)
            # axg.set_zlim(-100, 100)

            # axg.set_xlim(-10, 20)
            # axg.set_ylim(-20, 20)
            # axg.set_zlim(-10, 10)

            axg.set_xlim(-10, 50)
            axg.set_ylim(-50, 50)
            axg.set_zlim(-50, 50)

            #plot vector norm arrows
            #ax.quiver(visData[:,0][freqRangeInds[::skip]], visData[:,1][freqRangeInds[::skip]], visData[:,2][freqRangeInds[::skip]], visData[:,10][freqRangeInds[::skip]], visData[:,11][freqRangeInds[::skip]], visData[:,12][freqRangeInds[::skip]], length = .04, pivot='tail', color='g')
            ##plot B mag field lines upstream
            #Bnorms = np.sqrt(v0isData[:,6][freqRangeInds[::skip]]**2 + visData[:,7][freqRangeInds[::skip]]**2 + visData[:,8][freqRangeInds[::skip]]**2)
            #ax.quiver(visData[:,0][freqRangeInds[::skip]], visData[:,1][freqRangeInds[::skip]], visData[:,2][freqRangeInds[::skip]], visData[:,6][freqRangeInds[::skip]]/Bnorms, visData[:,7][freqRangeInds[::skip]]/Bnorms, visData[:,8][freqRangeInds[::skip]]/Bnorms, length = .04, pivot='tail')


            axg.set_xlabel('X Solar Radii')
            axg.set_ylabel('Y Solar Radii')
            axg.set_zlabel('Z Solar Radii')

            plt.title(str(mode) + ' plot, time ' + str(time) + '\n'  +label)

            figg.subplots_adjust(right=0.8)
            cbar_axg = figg.add_axes([0.85, 0.15, 0.05, 0.7])

            starts = array(map(int, logspace(log10(10.), log10(4095.), 64)))
            starts[:9] = [0, 2, 4, 6, 8, 11, 14, 17, 20] #better samples at low end here
            stepFreqs = newFreqs[starts]

            stepFreqs = stepFreqs/1000. #to mhz

            boundsg = np.append(0, stepFreqs)
            normg = mpl.colors.BoundaryNorm(boundsg, cmapg.N)
            cbg = matplotlib.colorbar.ColorbarBase(cbar_axg, cmap=cmapg, norm=normg, spacing='uniform', ticks=boundsg[::4], boundaries=boundsg, format='%2.1f')
            cbar_axg.set_ylabel('Observing Frequencies (MHz)', size=12)

            #fileName = 'Entropy_3D%3f-%3fMHz%s.png'%(freqlo, freqhi, datFile)
            fileName = '3D'+str(mode) + 'bitInd' + str(bitInd) + '-time' + str(time) + '.png'

            figg.savefig(fileName)
            plt.close()

            ##2D
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            # plt.title(str(mode) + ' plot, time ' + str(time) + '\n'  +label)
            plt.title(str(mode) + ' plot, time ' + str(time) + ' '  +label)

            fig1.subplots_adjust(right=0.8)
            cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
            cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmapg, norm=normg, spacing='uniform', ticks=boundsg[::4], boundaries=boundsg, format='%2.1f')
            cbar_ax.set_ylabel('Observing Frequencies (MHz)', size=12)

            p1=ax.scatter(visData[:,1][allGoodInds], visData[:,2][allGoodInds], c=freqColors)


            # ax.set_xlim(-150, 150)
            # ax.set_ylim(-150, 150)

            # ax.set_xlim(-20, 20)
            # ax.set_ylim(-20, 20)

            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)

            ax.set_xlabel('X Solar Radii')
            ax.set_ylabel('Y Solar Radii')


            sun = Ellipse(xy=(0, 0),
                        width=2.0, height=2.0,# figure = fig,
                        angle=0., fill = True)
            sun.set_facecolor('y')
            #ell.set_hatch('x')
            ax.add_artist(sun)

            #ax.set_xlabel('X Degrees from Sun')
            #ax.set_ylabel('Y Degrees from Sun')

            fileName = '2D'+str(mode) + 'bitInd' + str(bitInd) + '-time' + str(time) + '.png'

            fig1.savefig(fileName)

            plt.close()


            ##2D

            #end snapshot



    # z = my_interp(X, Y, cutData.T, x, y, spn=2)

    #im = ax1.imshow(cutData.T, interpolation='none')
    print(shape(Xsim), shape(Ysim), shape(cutDataCol1))

    #create last figure here with data we've been building up over time

    #from anal7.py, this creates labeled 2d histograms
    fig, ax1 = plt.subplots()#figsize=(16,8))


    # ax1.set_xlim((-0.5, len(times) - .5))
    # ax1.set_xticks(range(len(times)))
    # ax1.set_xticklabels(map(str, map(int, times)))

    ax1.set_title('Synthetic Spectra from ' + str(label), fontsize=18)

    ax1.set_xlim((min(times), max(times)))
    # ax1.xlim((X.min(), X.max()))
    ax1.set_ylim((min(freqs), max(freqs)))



    # ax1.set_xlabel('Minutes past 17:20 on 05/13/2005', fontsize=18)

    ax1.set_xlabel('Minutes past 16:47 on 05/13/2005', fontsize=18)

    ax1.set_ylabel('Frequency (kHz)', fontsize=18)
    ax1.set_yscale("log")


    im = ax1.pcolormesh(Xsim, Ysim, cutDataCol1.T, cmap = cm)#, interpolation='none')
    # im = ax1.imshow(cutDataCol1.T, cmap=cm)
    # cbar = fig.colorbar(im, ax=ax1, cmap = cm)#'jet
    cb = fig.colorbar(im)
    cb.set_label('Column Sum Normalized Distributions of Points', fontsize=12)#, rotation=90, va="bottom")


    # Rotate the tick labels and set their alignment.
    # plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
    #           rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(times)):
    #     for j in range(len(lows)):
    #         if cutDataTot[i, j] > 5:
    #             text = ax1.text(i, j, "%d"%cutDataTot[i, j],
    #                             ha="center", va="center", color="w",fontsize = 5)
    #

    # fig.tight_layout()
    #show()

    fig.savefig('2dhistTimeFreq'+mode+str(bitInd)+'Sum.png')#, bbox_inches='tight', pad_inches=0.0) #, bbox_inches='tight', pad_inches=0.0)

    plt.close('all')

    #interpolate cutDataCol1 to Wind time candence, preserving col sum = 1
    #then multiply element wise with stencil (col max normalized) and sum to get metric score

    # maxScore = -1.
    # maxLead = -999.
    # maxSynthData = -1.

    #loop here over different offset or lead times for convTime conversion, choose maximum score over time around 60 Minutes
    # for leadTime in linspace(-30, 29, 60):#linspace(-40, 19, 60):
    #
    #     # newSynthData = my_interp(Xsim/60.+17. + 1./3., Ysim, cutDataCol1.T, X, Y, spn=5)
    #     # convTimes = times/60.+17. + 20./60. + leadTime/60.  #+ 1./3. #simulation starts at 17:20, and time is in hours
    #     #change to 16:47 onset????
    #     convTimes = times/60.+16. + 47./60. + leadTime/60.  #+
    #
    #
    #     #new interp away
    #     #https://stackoverflow.com/questions/26212122/python-extend-2-d-array-and-interpolate-the-missing-values
    #     valsSynth = np.reshape(cutDataCol1.T, (len(freqs)*len(times)))
    #     ptsSynth = np.array([[i,j] for i in freqs for j in convTimes] )
    #     # grid_x, grid_y = np.mgrid[freqs, X[0,:]]
    #     newSynthData = interpolate.griddata(ptsSynth, valsSynth, (Y, X), method='linear', fill_value=0.)
    #
    #     #renormalize column sums to 1
    #     sh = shape(newSynthData)
    #     iis, js = sh
    #     for jjs in range(js):
    #         sumcol = np.sum(newSynthData[:,jjs])
    #         if sumcol == 0.:
    #             continue
    #         newSynthData[:, jjs] = newSynthData[:, jjs]/sumcol
    #
    #
    #     prod = np.multiply(newSynthData, stencilNorm)
    #
    #     score = np.sum(prod)
    #
    #
    #
    #     if score >= maxScore:
    #         maxScore = score
    #         maxLead = leadTime
    #         maxSynthData = newSynthData.copy()
    #
    #
    # # print(score)
    # maxScores.append((maxScore, bitInd, maxLead))
    # print('Max score for this interpolated data is  %2.2f'%maxScore)
    # # print('Max score obtained with time shift of 17:20 + ' +str(maxLead) + ' minutes')
    # print('Max score obtained with time shift of 16:47 + ' +str(maxLead) + ' minutes')

    #save new interpolated spectra

    # fig, ax = subplots(figsize=(16,8))
    #
    # print(shape(X), shape(Y), shape(maxSynthData))
    #
    # # p = ax.pcolormesh(newX[:-23, :], newY[:-23, :], newDbsqrt[:-23, :], vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
    # p = ax.pcolormesh(X, Y, maxSynthData, vmin=0., vmax=max(amax(maxSynthData), 0.0000001),cmap=cm)#gist_ncar_r")
    # ax.set_yscale("log")
    # ylabel('kHz', fontsize=sizein)
    # # xlabel('Hours past '+'2012/01/19'+' midnight', fontsize=sizein)
    # xlabel('Hours past '+'2005/05/13'+' midnight', fontsize=sizein)
    # # title("Synthetic Data Interpolated to Wind Cadence, 17:20+Lag: " + str(maxLead) + " Score: %2.2f"%maxScore + '\n' + label, fontsize=sizein)
    # title("Synthetic Data Interpolated to Wind Cadence, 16:47+Lag: " + str(maxLead) + " Score: %2.2f"%maxScore + '\n' + label, fontsize=sizein)
    #
    # # xlim((newX.min(), newX.max()))
    # # ylim((min(freqs), max(freqs[:-23])))
    #
    # xlim((X.min(), X.max()))
    # ylim((min(freqs), max(freqs)))
    #
    # cb = fig.colorbar(p)
    #
    # cb.set_label("Fraction of points with given frequency", fontsize=sizein)
    #
    #
    # ax.plot(xs, lowsF, 'w--')
    # # ax.plot(xs, mids, 'w.')
    # ax.plot(xs, highsF, 'w--')
    #
    # fig.savefig('interpSynthetic'+str(mode)+str(bitInd)+'Max.png')#str(ind)+


    #make shift=0 version

    # convTimes = times/60.+17. + 20./60. #+ leadTime/60.  #+ 1./3. #simulation starts at 17:20, and time is in hours
    #change to 16:47 onset????
    convTimes = times/60.+16. + 47./60. # + leadTime/60.  #+

    #new interp away
    #https://stackoverflow.com/questions/26212122/python-extend-2-d-array-and-interpolate-the-missing-values
    # aa, bb = shape(cutDataCol1)
    valsSynth = np.reshape(cutDataCol1.T, (len(freqs)*len(times)))
    ptsSynth = np.array([[i,j] for i in freqs for j in convTimes] )
    # grid_x, grid_y = np.mgrid[freqs, X[0,:]]
    newSynthData = interpolate.griddata(ptsSynth, valsSynth, (Y, X), method='linear', fill_value=0.)

    #renormalize column sums to 1
    sh = shape(newSynthData)
    iis, js = sh
    for jjs in range(js):
        sumcol = np.sum(newSynthData[:,jjs])
        if sumcol == 0.:
            continue
        newSynthData[:, jjs] = newSynthData[:, jjs]/sumcol


    prod = np.multiply(newSynthData, stencilNorm)

    score = np.sum(prod)

    zeroScores.append((score, bitInd))

    # print('0 lag (17:20 start) score for this interpolated data is %2.2f'%score)
    print('0 lag (16:47 start) score for this interpolated data is %2.2f'%score)


    #save new interpolated spectra

    fig, ax = subplots(figsize=(16,8))

    # p = ax.pcolormesh(newX[:-23, :], newY[:-23, :], newDbsqrt[:-23, :], vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
    # p = ax.pcolormesh(X, Y, newSynthData, vmin=0., vmax=max(amax(newSynthData), .0000001),cmap=cm)#gist_ncar_r")
    # p = ax.pcolormesh(X, Y, newSynthData, vmin=0., vmax=.3,cmap=cm)#gist_ncar_r")
    # p = ax.pcolormesh(X, Y, newSynthData, vmin=0., vmax=.1,cmap=cm)#gist_ncar_r")
    p = ax.pcolormesh(X, Y, newSynthData, vmin=0., vmax=.05,cmap=cm) #amax(newSynthData)
    ax.set_yscale("log")
    ylabel('kHz', fontsize=sizein)
    # xlabel('Hours past '+'2012/01/19'+' midnight', fontsize=sizein)
    xlabel('Hours past '+'2005/05/13'+' midnight', fontsize=sizein)
    # title("Synthetic Data Interpolated to Wind Cadence, 17:20+Lag: " + str(0.) + " Score: %2.2f"%score + '\n' + label , fontsize=sizein)
    title("Synthetic Data Interpolated to Wind Cadence, Score: %2.2f"%score + '\n' + label , fontsize=sizein)


    xlim((X.min(), X.max()))
    ylim((min(freqs), max(freqs)))

    cb = fig.colorbar(p)

    cb.set_label("Fraction of points with given frequency", fontsize=sizein)


    ax.plot(xs, lowsF, 'w--')
    # ax.plot(xs, mids, 'w.')
    ax.plot(xs, highsF, 'w--')

    fig.savefig('interpSynthetic'+str(mode)+str(bitInd)+'.png')#str(ind)+

    if makeSnaps:
        fileList = ''
        for t in times:
            fileList += '3D'+str(mode) + 'bitInd' + str(bitInd) + '-time' + str(t) + '.png '

        returned_value = subprocess.call('convert -delay 30 -loop 1 '+ fileList+' '+ '3D'+str(mode) + 'bitInd' + str(bitInd) +'CUT.gif', shell=True)

        # returned_value = subprocess.call('rm *bitInd*png', shell=True)

        fileList = ''
        for t in times:
            fileList += '2D'+str(mode) + 'bitInd' + str(bitInd) + '-time' + str(t) + '.png '

        returned_value = subprocess.call('convert -delay 30 -loop 1 '+ fileList+' '+ '2D'+str(mode) + 'bitInd' + str(bitInd) +'CUT.gif', shell=True)

        # returned_value = subprocess.call('rm *bitInd*png', shell=True)


        # json.dump(maxScores.tolist(), open('maxScores.txt", 'w'))
        # json.dump(zeroScores.tolist(), open('zeroScores.txt", 'w'))


# print('max scores, max lead times (minutes) from 17:20')
# print('max scores, max lead times (minutes) from 16:47')
# print(maxScores)
# print(sorted(maxScores))
# print('scores from no lag adjustment')
# print(zeroScores)
# print(sorted(zeroScores))

# sortedMaxScores = sorted(maxScores, reverse=True)
# json.dump(sortedMaxScores, open('maxScores.txt', 'w'))

# sortedZeroScores = sorted(zeroScores, reverse=True)
# json.dump(sortedZeroScores, open('maxZeroScores.txt', 'w'))


#plot progression and speed
makeSpeedPlot = False
if makeSpeedPlot:
    maxrs = np.array(maxrs)
    meanrs = np.array(meanrs)
    meanxs = np.array(meanxs)
    meanys = np.array(meanys)
    meanzs = np.array(meanzs)
    medianrs = np.array(medianrs)

    speeds = np.zeros(len(maxrs))
    speedsmean = np.zeros(len(maxrs))
    speedsmeanx = np.zeros(len(maxrs))
    speedsmeany = np.zeros(len(maxrs))
    speedsmeanz = np.zeros(len(maxrs))
    speedsmed = np.zeros(len(maxrs))

    for i in range(1, len(speeds)):
        speeds[i] = rskm*(maxrs[i] - maxrs[i-1])/((times[i] - times[i-1])*60) #speeds in km/s
    speeds[0] = speeds[1]

    for i in range(1, len(speeds)):
        speedsmean[i] = rskm*(meanrs[i] - meanrs[i-1])/((times[i] - times[i-1])*60) #speeds in km/s
        speedsmeanx[i] = rskm*(meanxs[i] - meanxs[i-1])/((times[i] - times[i-1])*60) #speeds in km/s
        speedsmeany[i] = rskm*(meanys[i] - meanys[i-1])/((times[i] - times[i-1])*60) #speeds in km/s
        speedsmeanz[i] = rskm*(meanzs[i] - meanzs[i-1])/((times[i] - times[i-1])*60) #speeds in km/s

    speedsmean[0] = speedsmean[1]
    speedsmeanx[0] = speedsmeanx[1]
    speedsmeany[0] = speedsmeany[1]
    speedsmeanz[0] = speedsmeanz[1]

    for i in range(1, len(speeds)):
        speedsmed[i] = rskm*(medianrs[i] - medianrs[i-1])/((times[i] - times[i-1])*60) #speeds in km/s
    speedsmed[0] = speedsmed[1]

    fig, ax1 = plt.subplots()
    plt.margins(0.7, 0.7)
    ax1.tick_params(axis='x', pad=10)

    # ax1.set_xticks(range(1, numTrials+1)[::8])
    # labs = linspace(0, 25, 64)[::8].astype(str)
    # for i in range(len(labs)):
    #     labs[i] = labs[i][:4] #cut off decimal rep
    # ax1.set_xticklabels(labs)

    ax1.plot(times[:-9], maxrs[:-9], 'b*', label='Max R')
    ax1.plot(times[:-9], meanrs[:-9], 'b-', label='Mean R')
    # ax1.plot(times, medianrs, 'b--', label='Median R')
    ax1.set_xlabel('Time (minutes after 16:47)')
    ax1.set_ylabel(r'Maximum Distance of Shock ($R_S$)', color='b')
    ax1.tick_params('y', colors='b')



    ax2 = ax1.twinx()
    ax2.plot(times[:-9], speeds[:-9], 'r*', label='Max R')
    ax2.plot(times[:-9], speedsmean[:-9], 'r-', label='Mean R')
    # ax2.plot(times, speedsmed, 'r--', label='Median R')
    lab = ax2.set_ylabel('Speed of Shock (km/s)', color='r')
    ax2.tick_params('y', colors='r')
    #
    # ax2.set_xticks(range(1, numTrials+1)[::8])
    # # ax1.set_xticklabels(('1.25', '2.0', '5.0', '10.0', '20.0', '40.0'))
    # labs = linspace(0, 25, 64)[::8].astype(str)
    # for i in range(len(labs)):
    #     labs[i] = labs[i][:4]
    # ax2.set_xticklabels(labs)
    plt.title('Height and Speed of Simulated Shock')
    ax1.legend()
    fig.savefig('shockSpeedMaxMean.png', bbox_inches='tight')#str(ind)+
    plt.close('all')

    #xyz mean speed plot
    fig, ax1 = plt.subplots()
    # plt.margins(0.7, 0.7)
    ax1.tick_params(axis='x', pad=10)
    ax1.plot(times[:-9], speedsmeanx[:-9], 'r', label=r'Mean $V_x$')
    ax1.plot(times[:-9], speedsmeany[:-9], 'b', label=r'Mean $V_y$')
    ax1.plot(times[:-9], speedsmeanz[:-9], 'g', label=r'Mean $V_z$')
    # ax1.plot(times, meanrs, 'b*', label='Mean R')
    # ax1.plot(times, medianrs, 'b--', label='Median R')
    ax1.set_xlabel('Time (minutes after 16:47)')
    ax1.set_ylabel(r'Speed of Shock (km/s)')
    ax1.tick_params('y')
    plt.title('Velocity of Simulated Shock')
    ax1.legend(loc=1)
    fig.savefig('shockVelocity.png', bbox_inches='tight')#str(ind)+
    plt.close('all')

makeheatMap = False

#only if doing all 1600 zeroscores to fill up heatmap
if makeheatMap:
    print('scores from no lag adjustment')
    print(shape(zeroScores))
    print(zeroScores)
    json.dump(zeroScores, open('zeroScoresNEW.txt', 'w'))

    longLows = linspace(-60., 57., 40)
    longHighs = longLows + 3.0

    latLows = linspace(-60., 57., 40)
    latHighs = latLows + 3.0

    heatMapData = zeros((40,40))
    maxii = min(1600, len(zeroScores))
    for ii in range(maxii):
        heatMapData[ii/40, ii%40] = zeroScores[ii][0]

    # heatMapData = np.flipud(heatMapData) #fix upside down from starting at -60

    #ind increases along constant lines of latitude, so /ind for lat, %ind for long

    xs = longLows.copy()
    ys = latLows.copy()

    XX, YY = meshgrid(xs, ys)


    fig, ax1 = plt.subplots()#figsize=(16,8))

    ax1.set_title('Similarity Scores of Longitude/Latitude \n Synthtic Spectra Cuts', fontsize=18)

    ax1.set_xlim((min(xs), max(xs)))
    # ax1.xlim((X.min(), X.max()))
    ax1.set_ylim((min(ys), max(ys)))

    ax1.set_xlabel('Longitude (degrees)', fontsize=18)

    ax1.set_ylabel('Latitude (degrees)', fontsize=18)


    im = ax1.pcolormesh(XX, YY, heatMapData)#, interpolation='none')
    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
    cbar.set_label('Similarity Compared to Stencil', fontsize=12)#, rotation=90, va="bottom")

    fig.tight_layout()
    #show()

    fig.savefig('NEWlongLatHeatmaps.png')#, bbox_inches='tight') #, bbox_inches='tight', pad_inches=0.0)

    plt.close('all')



# visDataArr =  json.load(open(datFile+"data.txt", 'r'))
