'''#Alex Hegedus
#https://solar-radio.gsfc.nasa.gov/wind/one_minute_doc.html
#all arrays are 256x1441
#All values are in terms of ratio to background and the background values are listed in position 1441 in microvolts per root Hz.
#create frequency selection figure from Wind data

This is another work horse script, instead of calculating a scored synthetic spectra, it creates
heatmaps of chosen variables over the surface of the event for a given time
'''

import matplotlib
matplotlib.use('Agg')

from pylab import *
from scipy.io import readsav
import os

import subprocess

from matplotlib.colors import LogNorm

import matplotlib.colors as colors

from scipy.stats import norm as normVar

from scipy import interpolate
from scipy.interpolate import UnivariateSpline

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

newX = newX[:, start:start+totMins:fact]
newY = newY[:, start:start+totMins:fact]



vminp = -.9#-1.15
vmaxp = 5#12. #5.#10.

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



cm =LinearSegmentedColormap.from_list('my_list', colors, N=150)
# cm = colors.ListedColormap(cpool[0:15], 'indexed')



fig, ax = subplots(figsize=(16,8))

# p = ax.pcolormesh(newX[:-23, :], newY[:-23, :], newDbsqrt[:-23, :], vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
p = ax.pcolormesh(X, Y, allDataDbsqrt, vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
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

cb.set_label("dB of sqrt(Intensity/Background)", fontsize=sizein)



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
p = ax.pcolormesh(X, Y, allDataDbsqrt, vmin=0., vmax=vmaxp,cmap=cm)#gist_ncar_r")
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

cb.set_label("dB of sqrt(Intensity/Background)", fontsize=sizein)


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

    for j in range(midFreqInd+1, highFreqInd+1):
        stencil[j, ii] = normVar.pdf(j, loc=midFreqInd, scale=sigup)
        stencilNorm[j, ii] = normVar.pdf(j, loc=midFreqInd, scale=sigup)/ normVar.pdf(midFreqInd, loc=midFreqInd, scale=sigup)



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
#

# cuts.append((17, 0, 90, 'Shock Front Nose', 'IsoEntropy=4_', 'Front', 0))

#1600 bits, 3x3 degrees from -60 to 60
# linspace(13*40, 20*40, 60)  #gets a range of good and bad bits for inspection, range(1, 1601) gets all
# for i in linspace(13*40+1, 20*40+1, 60): #range(1, 1601): #linspace(13*40, 20*40, 60):#[0]:#[825, 826]: #[820, 824, 816, 700, 704, 696, 940, 944, 936]:#range(1, 1601):#5, 10, 15]:#range(0, 1):
#     cuts.append((17, 0, 90, 'Entropy Shock Front', 'IsoEntropy=4_', 'FrontBit', int(i)))
#
#
# cuts.append((17, 0, 90, 'Density Shocked Region', 'isoNe=3.5_', 'FlankBit', 0))
#
# cuts.append((17, 0, 90, 'Current Sheet Region', 'iso_Tp=3.5MK_', 'Sheet', 0))




# #radial velocity
# cuts.append((3, 0, 9000, '(Upstream) Velocity Magnitude', 'IsoEntropy=4_', 'FrontBitVAR', 20))
# cuts.append((3, 0, 9000, '(Upstream) Velocity Magnitude', 'IsoEntropy=4_', 'FrontBitVAR', 40))
# cuts.append((3, 0, 9000, '(Upstream) Veloticy Magnitude', 'IsoEntropy=4_', 'FrontBitVAR', 90))
# cuts.append((3, 0, 9000, '(Upstream) Total Velocity', 'IsoEntropy=4_', 'FrontBitVAR', 120))

#
# cuts.append((6, 0, 9000, '(Upstream) Magnetic Field Magnitude', 'IsoEntropy=4_', 'FrontBitVAR', 20))
# cuts.append((6, 0, 9000, '(Upstream) Magnetic Field Magnitude', 'IsoEntropy=4_', 'FrontBitVAR', 40))
# cuts.append((6, 0, 9000, '(Upstream) Magnetic Field Magnitude', 'IsoEntropy=4_', 'FrontBitVAR', 90))
# # cuts.append((6, 0, 9000, '(Upstream) Magnetic Field Magnitude', 'IsoEntropy=4_', 'FrontBitVAR', 120))
#
#
# cuts.append((7, 0, 9000, '(Upstream) Alfven Speed', 'IsoEntropy=4_', 'FrontBitVAR', 20))
# cuts.append((7, 0, 9000, '(Upstream) Alfven Speed', 'IsoEntropy=4_', 'FrontBitVAR', 40))
# cuts.append((7, 0, 9000, '(Upstream) Alfven Speed', 'IsoEntropy=4_', 'FrontBitVAR', 90))
# # cuts.append((7, 0, 9000, '(Upstream) Alfven Speed', 'IsoEntropy=4_', 'FrontBitVAR', 120))
# #
# # #
# cuts.append((9, 0, 90, r'Upstream Plasma Frequency', 'IsoEntropy=4_', 'FrontBitVAR', 20))
# cuts.append((9, 0, 90, r'Upstream Plasma Frequency', 'IsoEntropy=4_', 'FrontBitVAR', 40))
# cuts.append((9, 0, 90, r'Upstream Plasma Frequency', 'IsoEntropy=4_', 'FrontBitVAR', 90))
# # cuts.append((9, 0, 90, r'Upstream Plasma Frequency', 'IsoEntropy=4_', 'FrontBitVAR', 120))
# # #
# #
# cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', 'IsoEntropy=4_', 'FrontBitVAR', 20))
# cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', 'IsoEntropy=4_', 'FrontBitVAR', 40))
# cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', 'IsoEntropy=4_', 'FrontBitVAR', 90))
# # cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', 'IsoEntropy=4_', 'FrontBitVAR', 120))
# #
# cuts.append((17, 0, 90, r'$\Theta_{Bn}$', 'IsoEntropy=4_', 'FrontBitVAR', 20))
# cuts.append((17, 0, 90, r'$\Theta_{Bn}$', 'IsoEntropy=4_', 'FrontBitVAR', 40))
# cuts.append((17, 0, 90, r'$\Theta_{Bn}$', 'IsoEntropy=4_', 'FrontBitVAR', 90))
# # cuts.append((17, 0, 90, r'$\Theta_{Bn}$', 'IsoEntropy=4_', 'FrontBitVAR', 120))
#
# # cuts.append((16, 2000., 100000, 'de Hoffmann-Teller Velocity 2000 km/s +', 'IsoEntropy=4_', 'FrontBitVAR', 20))
#
# #
# # cuts.append((17, 0, 90, 'Theta_{Bn}', 'isoNe=3.5_', 'FrontBitVAR', 20))
# # cuts.append((17, 0, 90, 'Theta_{Bn}', 'isoNe=3.5_', 'FrontBitVAR', 40))
# # cuts.append((17, 0, 90, 'Theta_{Bn}', 'isoNe=3.5_', 'FrontBitVAR', 120))
#
# #electron density
# cuts.append((23, 0, 90, r'Upstream $n_e$ / cc', 'IsoEntropy=4_', 'FrontBitVAR', 20))
# cuts.append((23, 0, 90, r'Upstream $n_e$ / cc', 'IsoEntropy=4_', 'FrontBitVAR', 40))
# cuts.append((23, 0, 90, r'Upstream $n_e$ / cc', 'IsoEntropy=4_', 'FrontBitVAR', 90))
# # cuts.append((23, 0, 90, r'Upstream $n_e$ / cc', 'IsoEntropy=4_', 'FrontBitVAR', 120))
#
# cuts.append((16, 2250., 99999, 'de Hoffmann-Teller Velocity 2250+', 'IsoEntropy=4_', 'FrontBitVAR', 8))
# cuts.append((16, 2250., 99999, 'de Hoffmann-Teller Velocity 2250+', 'IsoEntropy=4_', 'FrontBitVAR', 10))
# cuts.append((16, 1850., 99999, 'de Hoffmann-Teller Velocity 2250+', 'IsoEntropy=4_', 'FrontBitVAR', 12))
# cuts.append((16, 1850., 99999, 'de Hoffmann-Teller Velocity 2250+', 'IsoEntropy=4_', 'FrontBitVAR', 14))
# cuts.append((16, 1850., 99999, 'de Hoffmann-Teller Velocity 2250+', 'IsoEntropy=4_', 'FrontBitVAR', 16))


cuts.append((9, 0, 90, r'Upstream Plasma Frequency', '2005CME_iso_Machms=3_', 'FrontBitVAR', 10))
cuts.append((9, 0, 90, r'Upstream Plasma Frequency', '2005CME_iso_Machms=3_', 'FrontBitVAR', 20))
cuts.append((9, 0, 90, r'Upstream Plasma Frequency', '2005CME_iso_Machms=3_', 'FrontBitVAR', 30))
cuts.append((9, 0, 90, r'Upstream Plasma Frequency', '2005CME_iso_Machms=3_', 'FrontBitVAR', 40))
cuts.append((9, 0, 90, r'Upstream Plasma Frequency', '2005CME_iso_Machms=3_', 'FrontBitVAR', 60))
cuts.append((9, 0, 90, r'Upstream Plasma Frequency', '2005CME_iso_Machms=3_', 'FrontBitVAR', 90))

cuts.append((36, 0, 90, r'CME Magnetosonic Mach #', '2005CME_iso_Machms=3_', 'FrontBitVAR', 10))
cuts.append((36, 0, 90, r'CME Magnetosonic Mach #', '2005CME_iso_Machms=3_', 'FrontBitVAR', 20))
cuts.append((36, 0, 90, r'CME Magnetosonic Mach #', '2005CME_iso_Machms=3_', 'FrontBitVAR', 30))
cuts.append((36, 0, 90, r'CME Magnetosonic Mach #', '2005CME_iso_Machms=3_', 'FrontBitVAR', 40))
cuts.append((36, 0, 90, r'CME Magnetosonic Mach #', '2005CME_iso_Machms=3_', 'FrontBitVAR', 60))
cuts.append((36, 0, 90, r'CME Magnetosonic Mach #', '2005CME_iso_Machms=3_', 'FrontBitVAR', 90))

cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', '2005CME_iso_Machms=3_', 'FrontBitVAR', 10))
cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', '2005CME_iso_Machms=3_', 'FrontBitVAR', 20))
cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', '2005CME_iso_Machms=3_', 'FrontBitVAR', 30))
cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', '2005CME_iso_Machms=3_', 'FrontBitVAR', 40))
cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', '2005CME_iso_Machms=3_', 'FrontBitVAR', 60))
cuts.append((16, -9999, 99999, 'de Hoffmann-Teller Velocity', '2005CME_iso_Machms=3_', 'FrontBitVAR', 90))


cuts.append((17, 0, 90, r'$\Theta_{Bn}$', '2005CME_iso_Machms=3_', 'FrontBitVAR', 10))
cuts.append((17, 0, 90, r'$\Theta_{Bn}$', '2005CME_iso_Machms=3_', 'FrontBitVAR', 20))
cuts.append((17, 0, 90, r'$\Theta_{Bn}$', '2005CME_iso_Machms=3_', 'FrontBitVAR', 30))
cuts.append((17, 0, 90, r'$\Theta_{Bn}$', '2005CME_iso_Machms=3_', 'FrontBitVAR', 40))
cuts.append((17, 0, 90, r'$\Theta_{Bn}$', '2005CME_iso_Machms=3_', 'FrontBitVAR', 60))
cuts.append((17, 0, 90, r'$\Theta_{Bn}$', '2005CME_iso_Machms=3_', 'FrontBitVAR', 90))

txtPrePath = "/mnt/LinuxData/CME_Data/"

maxScores = []

zeroScores = []

# bitInd in this script takes the place of the minute of data to plot a variable in a heat map over the surface

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

    for t in range(len(times)):


        plt.close('all')

        time = times[t]

        if time != bitInd:
            continue


        datFile = datPre + str(time) +'min.dat'

        if datFile[:4] != '2005':
            if time < 10:
                datFile = datPre + '00' + str(time) +'min.dat'
            elif time < 100:
                datFile = datPre + '0' + str(time) +'min.dat'

        dataPath = txtPrePath+datFile+"dataTBN.txt"
        if not os.path.exists(dataPath):
            continue

        data =  json.load(open(dataPath, 'r'))

        if time == 180:
            dataIH =  json.load(open(txtPrePath + "IsoEntropy=4_180minIH.datdata.txt", 'r'))
            dataSC =  json.load(open(txtPrePath + "IsoEntropy=4_180minSC.datdata.txt", 'r'))
            data = np.append(dataIH, dataSC, axis=0)
            data = data.tolist()

            data =  json.load(open(txtPrePath + "IsoEntropy=4_180minSC.datdata.txt", 'r'))

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


        #fix thetaBN to 0-90 instead of 0-180
        for i in range(np.shape(data)[0]):
            if data[i, 17] > 90.:
                data[i, 17] = 180. - data[i, 17]

        deg2D = data[:, 1:3]
        visData = data.copy()

        #do cut from 4panel
        if mode == 'Front':

            ###alternate version, using better cut for artifact generally used in frontbit

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
            longs = np.arctan2(visData[:, 1], visData[:, 0])/np.pi*180.
            lats = np.arcsin(visData[:, 2]/ r3s)/np.pi*180.

            #remove close in artifact
            maxr = max(r3s)
            print('max r out projected is ' + str(maxr))

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
            # smallR = np.where(np.logical_or(r3s < 2., np.logical_and(r3s < maxr*.7, np.logical_and(lats > 20., lats < 70.))))[0]

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

            largeLong = np.where(r3s < 2.0 )[0]

            visData = np.delete(visData, largeLong, axis=0)

            data = visData.copy()


        #do cut from 4panel
        #
        if mode == 'FrontBit':

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
                if len(visData) != 0:
                    r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)

                    tinymax = max(r3s)

                    smallData = np.where(r3s < 0.8*tinymax)
                    visData = np.delete(visData, smallData, axis=0)

                data = visData.copy()

                strbit = '\n Longitude %2.0f - %2.0f degrees, Latitude %2.0f - %2.0f'%(longLows[bitInd%40], longHighs[bitInd%40], latLows[bitInd/40], latHighs[bitInd/40])

                bitInd = bitInd+1

                label = 'Entropy Shock Flank Bit Ind ' + str(bitInd) + strbit



        #use bitInd to make variable histogram data on certain time
        if mode == 'FrontBitVAR':

            #make plot only on certain time of simulation 40 and 120 for now, must be time in snapshot cadence, no interpolating
            if time == bitInd:

                if datFile[:4] != '2005':
                    #additional hi mag cut, artifact?
                    bmag = np.sqrt(visData[:, 6]**2 + visData[:, 7]**2 + visData[:, 8]**2)
                    # bmag = np.sqrt(tempData[:, 6]**2 + tempData[:, 7]**2 + tempData[:, 8]**2)
                    maxmag = max(bmag)
                    bigmag = np.where(bmag > 0.1*maxmag)
                    visData = np.delete(visData, bigmag, axis=0)

                    #only top of
                    r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)
                    tinymax = max(r3s)
                    smallData = np.where(r3s < 0.7*tinymax)
                    visData = np.delete(visData, smallData, axis=0)


                ####### calc variables
                azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
                r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
                r3s = sqrt(visData[:, 0]**2 + visData[:, 1]**2 + visData[:, 2]**2)

                longs = np.arctan2(visData[:, 1], visData[:, 0])/np.pi*180.
                lats = np.arcsin(visData[:, 2]/ r3s)/np.pi*180.

                bmag = np.sqrt(data[:, 6]**2 + data[:, 7]**2 + data[:, 8]**2)

                #remove close in artifact
                maxr = max(r2s)
                print('max r out projected is ' + str(maxr))


                enhanceFactor = 2

                longLows = linspace(-60., 57., 40*enhanceFactor)
                longHighs = longLows + 3.0/enhanceFactor

                latLows = linspace(-60., 57., 40*enhanceFactor)
                latHighs = latLows + 3.0/enhanceFactor

                varScores = zeros((40*enhanceFactor,40*enhanceFactor))
                varScoresStd = zeros((40*enhanceFactor,40*enhanceFactor))

                #fill out 2d hist map of variable at this time in shock cut
                for q in range(1600*enhanceFactor*enhanceFactor):



                #arranged so lines of constant latitude are done in succession
                    badAzi = np.where(np.logical_or(np.logical_or(lats < latLows[q/(40*enhanceFactor)], lats > latHighs[q/(40*enhanceFactor)]), np.logical_or(longs < longLows[q%(40*enhanceFactor)], longs > longHighs[q%(40*enhanceFactor)])))[0]

                    tempData = np.delete(visData, badAzi, axis=0)
                    # tempData = np.delete(tempData, badAzi, axis=0)



                    #this if cut makes the difference between zeroscores 1 (commmented) and zeroscores2 (uncommented) zeroscores 2 is better, smoother heatmap
                    if len(tempData) != 0:
                        r3s = sqrt(tempData[:, 0]**2 + tempData[:, 1]**2 + tempData[:, 2]**2)

                        tinymax = max(r3s)

                        # smallData = np.where(r3s < 0.8*tinymax)
                        # tempData = np.delete(tempData, smallData, axis=0)

                        data = tempData.copy()

                        if varInd == 6:

                            bmag = np.sqrt(data[:, varInd]**2 + data[:, varInd+1]**2 + data[:, varInd+2]**2)
                            varScores[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.mean(bmag)
                            varScoresStd[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.std(bmag)

                        #use this for alfven speed
                        elif varInd == 7:
                            mu0 = 1.2566e-6
                            bmag = np.sqrt(data[:, varInd]**2 + data[:, varInd+1]**2 + data[:, varInd-1]**2)
                            alfvenspeed = bmag/np.sqrt(mu0*data[:, 24])
                            varScores[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.mean(alfvenspeed)
                            varScoresStd[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.std(alfvenspeed)

                        elif varInd == 3:
                            umag = np.sqrt(data[:, varInd]**2 + data[:, varInd+1]**2 + data[:, varInd+2]**2)
                            varScores[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.mean(umag)
                            varScoresStd[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.std(umag)


                        else:
                            varScores[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.mean(data[:, varInd])
                            varScoresStd[q/(40*enhanceFactor), q%(40*enhanceFactor)] = np.std(data[:, varInd])#average of variable

                    # heatMapData = np.flipud(heatMapData) #fix upside down from starting at -60

                    #ind increases along constant lines of latitude, so /ind for lat, %ind for long

                xs = longLows.copy()
                ys = latLows.copy()

                XX, YY = meshgrid(xs, ys)


                fig, ax1 = plt.subplots()#figsize=(16,8))

                ax1.set_title('Mean of ' + label + '\n after ' + str(time) + ' minutes', fontsize=18)

                ax1.set_xlim((min(xs), max(xs)))
                # ax1.xlim((X.min(), X.max()))
                ax1.set_ylim((min(ys), max(ys)))

                ax1.set_xlabel('Longitude (degrees)', fontsize=18)

                ax1.set_ylabel('Latitude (degrees)', fontsize=18)

                # XX = XX[5:-5, 5:-5]
                # YY = YY[5:-5, 5:-5]
                # varScores = varScores[5:-5, 5:-5]


                # if varInd == 6 and bitInd == 40:
                #     im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=.05)#
                #     cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                #     cbar.set_label('Gauss', fontsize=18)

                if varInd == 6:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=.3)#amax(varScores))#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label('Gauss', fontsize=18)
                    # cbar.set_norm(norm=matplotlib.colors.LogNorm())


                elif varInd == 3 and bitInd == 40:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=480.)#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label('km/s', fontsize=18)#, rotation=90, va="bottom")

                elif varInd == 17:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=amax(varScores))#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label(r'$\Theta_{Bn}$', fontsize=18)#, rotation=90, va="bottom")

                elif varInd == 16:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=3000.)#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label('km/s', fontsize=18)#, rotation=90, va="bottom")

                elif varInd == 9:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=min(4., amax(varScores)))#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label('MHz', fontsize=18)#, rotation=90, va="bottom")

                elif varInd == 23:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=amax(varScores))#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label(r'$n_e$ /cc', fontsize=18)#, rotation=90, va="bottom")

                elif varInd == 35:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=amax(varScores))#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label(r'Magnetosonic Mach #', fontsize=18)#, rotation=90, va="bottom")

                else:
                    im = ax1.pcolormesh(XX, YY, varScores, vmin=amin(varScores), vmax=amax(varScores))#
                    cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                    cbar.set_label('km/s', fontsize=18)#, rotation=90, va="bottom")

                # fig.tight_layout()
                #show()

                fig.savefig('Entvar'+str(varInd)+'Heatmaps'+str(time)+'.png')#, bbox_inches='tight') #, bbox_inches='tight', pad_inches=0.0)

                plt.close('all')


                ##now std dev
                fig, ax1 = plt.subplots()#figsize=(16,8))

                ax1.set_title('Standard Deviation of ' + label + '\n after ' + str(time) + ' minutes', fontsize=18)

                ax1.set_xlim((min(xs), max(xs)))
                # ax1.xlim((X.min(), X.max()))
                ax1.set_ylim((min(ys), max(ys)))

                ax1.set_xlabel('Longitude (degrees)', fontsize=18)

                ax1.set_ylabel('Latitude (degrees)', fontsize=18)


                im = ax1.pcolormesh(XX, YY, varScoresStd, vmin=amin(varScoresStd), vmax=amax(varScoresStd))
                cbar = fig.colorbar(im, ax=ax1, cmap = 'jet')
                # cbar.set_label('Similarity Compared to Stencil', fontsize=12)#, rotation=90, va="bottom")

                # fig.tight_layout()
                #show()

                fig.savefig('Entvar'+str(varInd)+'Heatmapsstddev'+str(time)+'.png')#, bbox_inches='tight') #, bbox_inches='tight', pad_inches=0.0)

                plt.close('all')

                #make 2D POS data cut png
                freqscat = np.array([[.25, .5], [.35, .375], [.525, .25], [.9, .175], [1.5, .125] , [4.5, .075], [15., .05]])
                splscat = UnivariateSpline(freqscat[:, 0], freqscat[:, 1], k=1, s=0)

                freqData = visData[:, 9]#*1000 #mhz to khz
                varIndData = visData[:, varInd]
                deg2D = visData[:, 1:3]
                dRs = .53/2 #degrees of Rs in sky

                for freqlo in [6., 10.]:
                    freqhi = freqlo + .100

                    freqRangeInds = np.where(np.logical_and(np.logical_and(freqData > freqlo, freqData < freqhi), np.logical_and(varIndData >= varLow, varIndData <= varHigh)))[0]
                    z = np.zeros((1024, 1024))
                    interpSize = 256
                    scatWidthDeg = max(splscat(freqlo), .05)

                    #one pix  = 1 arcmin, 60 pix is 1 deg
                    g1 = makeGaussian(1024, fwhm =  int(np.round(60.*scatWidthDeg)))#10*freqlo)

                    #one pix = 4 arcmin, 15 pix is 1 deg
                    gsmall = makeGaussian(interpSize, fwhm =  int(np.round(15.*scatWidthDeg)))#10*freqlo)

                    smallz = np.zeros((interpSize, interpSize))
                    smallxcoor = np.linspace(-512*1./60, 512*1./60, interpSize) #4 armin with 256 pix
                    smallycoor = np.linspace(-512*1./60, 512*1./60, interpSize)

                    xcoor = np.linspace(-512*1./60, 512*1./60, 1024)# so each pix is 1 arcmin
                    ycoor = np.linspace(-512*1./60, 512*1./60, 1024)
                    xx, yy = np.meshgrid(xcoor, ycoor)

                    for ind in range(len(freqRangeInds)):
                        indx = np.where(abs(deg2D[:,0][freqRangeInds][ind] - smallxcoor) == min(abs(deg2D[:,0][freqRangeInds][ind] - smallxcoor)))[0][0]
                        indy = np.where(abs(deg2D[:,1][freqRangeInds][ind] - smallycoor) == min(abs(deg2D[:,1][freqRangeInds][ind] - smallycoor)))[0][0]
                        smallz[indx][indy] = 1.
                        indx = np.where(abs(deg2D[:,0][freqRangeInds][ind] - xcoor) == min(abs(deg2D[:,0][freqRangeInds][ind] - xcoor)))[0][0]
                        indy = np.where(abs(deg2D[:,1][freqRangeInds][ind] - xcoor) == min(abs(deg2D[:,1][freqRangeInds][ind] - ycoor)))[0][0]
                        z[indx][indy] = 1.


                    fInterp = interpolate.interp2d(smallxcoor, smallycoor, smallz)

                    gaussfft = np.fft.fft2(np.fft.fftshift(g1))
                    gaussfftsmall = np.fft.fft2(np.fft.fftshift(gsmall))

                    znewInterp = fInterp(xcoor, ycoor)


                    znewInterp = np.flipud(znewInterp.T)
                    znewInterpfft = np.fft.fft2(np.fft.fftshift(znewInterp))
                    znewInterpGauss = (np.fft.fftshift(np.fft.ifft2(znewInterpfft*gaussfft))).real

                    znewFull = np.flipud(z.T)
                    znewFullfft = np.fft.fft2(np.fft.fftshift(znewFull))
                    znewFullGauss = (np.fft.fftshift(np.fft.ifft2(znewFullfft*gaussfft))).real

                    zsmall = np.flipud(smallz.T)
                    zsmallfft = np.fft.fft2(np.fft.fftshift(zsmall))
                    zsmallGauss = (np.fft.fftshift(np.fft.ifft2(zsmallfft*gaussfftsmall))).real

                    # plt.imsave('ARR'+str(freqlo) + '-' + str(freqhi) + 'MHz'+str(datFile)+'Gauss.png', znewInterpGauss)
                    # plt.imsave('ARRsmall'+str(freqlo) + '-' + str(freqhi) + 'MHz'+str(datFile)+'Gauss.png', zsmallGauss)



            else:
                continue





        if mode == 'Flank':

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)

            maxr = max(r2s)


            smallAzi = np.where(azimuths > -30.)[0]

            visData = np.delete(visData, smallAzi, axis=0)

            deg2D = visData[:, 1:3]

            azimuths = np.arctan2(deg2D[:,1], deg2D[:, 0])/np.pi*180.
            r2s = sqrt(deg2D[:,0]**2 + deg2D[:, 1]**2)
            if len(r2s) == 0:
                continue


            maxr = max(r2s)
            print('max r out projected is ' + str(maxr))
            smallR = np.where(r2s < maxr/2.)[0]

            visData = np.delete(visData, smallR, axis=0)

            data = visData.copy


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
