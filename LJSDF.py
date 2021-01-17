#!/usr/bin/env python
# coding: utf-8

'''
LENNARD-JONES SOFT-DISKS FLUID LEARNING (SDFL).
Author: Luca Zammataro, 2020.
'''


import pandas as pd
import math
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from PIL import Image
import glob
import moviepy.editor as mp
from datetime import datetime
import time
import sys



# CLASS DEFINITIONS

class Mol():
    def __init__(self, r, rv, ra):
        self.r = np.asarray([0.0, 0.0]) 
        self.rv = np.asarray([0.0, 0.0])
        self.ra = np.asarray([0.0, 0.0])
        
        
class Prop():
    def __init__(self, val, sum1, sum2 ):
        self.val=val
        self.sum1=sum1
        self.sum2=sum2 



# BASIC FUNCTIONS

# Sqr and Cube functions:

def Sqr(x):
    return (x * x) 

def Cube(x):
    return ((x) * (x) * (x))

  
# Randomness functions: 

def RandR():
    global randSeedP
    randSeedP = (randSeedP * IMUL + IADD) & MASK
    return (randSeedP * SCALE)

def VRand(p):
    s: float
    s = 2. * math.pi * RandR()
    p[0] = math.cos(s)
    p[1] = math.sin(s)
    return p


# Toroidal functions:

def VWrapAll(v):
    if v[0] >= 0.5 * region[0]:
        v[0] -= region[0]
    elif v[0] < -0.5 * region[0]:
        v[0] += region[0]
        
    if v[1] >= 0.5 * region[1]:
        v[1] -= region[1]
    elif v[1] < -0.5 * region[1]:
        v[1] += region[1]        

        
        
# Toroidal functions:

def VWrapAll_out(v):
    if v[0] >= 0.5 * region[0]:
        v[0] -= region[0]
    elif v[0] < -0.5 * region[0]:
        v[0] += region[0]
        
    if v[1] >= 0.5 * region[1]:
        v[1] -= region[1]
    elif v[1] < -0.5 * region[1]:
        v[1] += region[1]  
    
    return v

        
        
# This function updates coordinates taking care of periodic boundaries    
def ApplyBoundaryCond():
   
    for n in range(nMol):
        VWrapAll(mol[n].r)
        #print(n, mol[n].r)



# INITIALIZE COORDINATES.
# Here a simple square lattice (with the option of unequal edge lenghts) is used,
# so that each cell contains just one atom and the system is centered about the origin
def InitCoords():
    
    c = np.asarray([0.0, 0.0]) # Coords
    gap = np.divide(region, initUcell)
    n = 0
    for ny in range(0, int(initUcell[1])):
        for nx in range(0, int(initUcell[0])):
            
            
            mol[n].r = np.add(np.multiply(np.asarray([nx+0.5, ny+0.5]), gap), np.multiply(-0.5, region)) 
            coo_list.append(mol[n].r)
            n = n+1
            
            
# INITIALIZE VELOCITIES.
# The initial velocities of all the particles in the fluid simulation are set to a fixed mag-nitude (velMag), 
# which depends on the temperature. After assigning random velocity directions, 
# the velocities are adjusted to ensure that the center of mass is stationary.
# The function vRand serves as a source of uniformly distribuited random unit vectors.
def InitVels():
    
    global vSum
    vSum = np.zeros(vSum.shape)    
    
    for n in range(nMol):
        VRand(mol[n].rv)
        mol[n].rv = np.multiply(mol[n].rv, velMag)                
        vSum = np.add(vSum, mol[n].rv)


    for n in range(nMol):
        mol[n].rv = np.add(mol[n].rv, np.multiply((- 1.0 / nMol),  vSum))
        
        
         

    
# INITIALIZE ACCELERATIONS.
# The accelerations are initilized to zero
def InitAccels():
    for n in range(nMol):
        mol[n].ra = np.zeros(mol[n].ra.shape)




# Set parameters
def SetParams():

    global rCut
    global region
    global velMag # velocity magnitude
    
    rCut = math.pow(2., 1./6. * sigma)
    # Define the region
    region = np.multiply( 1./math.sqrt(density), initUcell)    

    #velocity magnitude depends on the temperature
    velMag = math.sqrt(NDIM * (1. -1. /nMol) * temperature)

        
# Setup Job
def SetupJob():
    
    global stepCount #  timestep counter 

    stepCount = 0 
    InitCoords()
    InitVels()
    InitAccels()
    AccumProps(0)



# FORCES COMPUTATION
'''
ComputeForces

ComputeForces is responsible for the interaction computations, and the interactions occur between pairs of atoms. 
The function implements the LJP, and calculates the accelerations and the forces for each pairs of atoms i and j 
located at ri and rj.
rCut = Limiting separation cutoff (rc), and it is: rCut = math.pow(2., 1./6.)
As r increases towards rCut, the force drops to 0.
Newton's third law inplies that fji = -fij, so each atom pair need only be examined once.
The amount of work is proportional to N^2.
'''

def ComputeForces():
    
    global virSum
    global uSum 
    fcVal = 0 #  The force that atom j exerts on atom i
 
    # rCut: Rc
    rrCut = Sqr(rCut)
    for n in range(nMol):
        mol[n].ra = np.zeros(mol[n].ra.shape)
        
    uSum = 0.
    virSum = 0.

    #n = 0
    for j1 in range(nMol-1):
        for j2 in range(j1+1, nMol):
            
            # Make DeltaRij: (sum of squared RJ1-RJ2)
            dr = np.subtract(mol[j1].r, mol[j2].r) # dr contains the delta between Rj1 and Rj2
            VWrapAll(dr) # toroidal function
            rr= (dr[0] * dr[0] + dr[1] * dr[1]) # dr2
            r= np.sqrt(rr) #dr

            
            # if dr2 < Rc^2 
            if (rr < rrCut):
                rri = sigma / rr                
                rri3 = Cube(rri)
                
                # Forces calculation by Lennard-Jones potential (original from Rapaport)
                #fcVal = 48. * rri3 * (rri3 - 0.5) * rri
                
                # Forces calculated with the completed Lennard-Jones.
                fcVal = 48 * epsilon * np.power(sigma, 12) / np.power(r, 13) - 24 * epsilon * np.power(sigma, 6) / np.power(r, 7) 

                # Update the accelerations multiplying force for DeltaRij
                mol[j1].ra = np.add(mol[j1].ra, np.multiply(fcVal, dr))
                mol[j2].ra = np.add(mol[j2].ra, np.multiply(-fcVal, dr))
                
                # Lennard-Jones potential (original from Rapaport)
                #uSum += 4. * rri3 * (rri3 - 1.) +1. 
                # The completed Lennard-Jones.
                uSum += 4 * epsilon * np.power(sigma/r, 12)/r - np.power(sigma/r, 6) # balanced              

                virSum += fcVal * rr
                

# INTEGRATION
'''
INTEGRATION OF COORDINATES AND VELOCITIES.
Integration of Equation of Motion uses a simple numerical techniques: the leapfrog method.
The method has excellent energy conservation properties.
LeapfrogStep integrates the coordinates and velocities. It appears twice in the listing of
SingleStep, with the argument part determinating which portion of the two-step leapfrog process
is to be performed:

vix(t + h/2) = vix(t) + (h/2)aix(t)
rix(t + h) = rix(t) + hvix (t + h/2)

'''
def LeapfrogStep(part):
    
    
    if part == 1:
        for n in range(nMol):
            mol[n].rv = np.add(mol[n].rv, np.multiply(0.5 * deltaT, mol[n].ra))            
            mol[n].r = np.add(mol[n].r, np.multiply(deltaT, mol[n].rv))                        
            
    else :
        for n in range(nMol):
            mol[n].rv = np.add(mol[n].rv, np.multiply(0.5 * deltaT, mol[n].ra))                        



# PROPERTIES MEASUREMENTS

def EvalProps():
    
    global vSum
    vvSum = 0.
    vSum = np.zeros(vSum.shape)
    
    global kinEnergy
    global uEnergy
    global totEnergy
    global pressure
    
    
    for n in range(nMol):
        vSum=np.add(vSum, mol[n].rv)
        vv= (mol[n].rv[0] * mol[n].rv[0] + mol[n].rv[1] * mol[n].rv[1])
        vvSum += vv
        
    kinEnergy.val = (0.5 * vvSum) / nMol
    uEnergy.val = (uSum / nMol)
    totEnergy.val = kinEnergy.val + (uSum / nMol)
    pressure.val = density * (vvSum + virSum) / (nMol * NDIM)
    
    
    
# AccumProps functions

def PropZero(v):
    v.sum1 = v.sum2 = 0.
    return v    
    
def PropAccum(v):
    v.sum1 += v.val
    v.sum2 += Sqr(v.val)
    return v    
    
def PropAvg(v, n):
    v.sum1 /= n
    v.sum2 = math.sqrt(max(v.sum2 / n - Sqr(v.sum1), 0.)) 
    return v    
    

# AccumProps: collects results of the measurements and evaluates means and standard deviation
def AccumProps(icode):
    
    
    if icode == 0:
        PropZero(totEnergy)
        PropZero(kinEnergy)
        PropZero(uEnergy)
        PropZero(pressure) 
    if icode == 1:
        PropAccum(totEnergy)
        PropAccum(kinEnergy)
        PropAccum(uEnergy)
        PropAccum(pressure)    
    if icode == 2:
        PropAvg(totEnergy, stepAvg)
        PropAvg(kinEnergy, stepAvg)
        PropAvg(uEnergy, stepAvg)
        PropAvg(pressure, stepAvg) 



# OUTPUT FUNCTIONS:

def plotMol(typeOfData, workdir, n):
    
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    

    if(df_params.loc[df_params['parameter']=='mode'].values[0][1]!='simulation'):
        A_pos = int(df_params.loc[df_params['parameter']=='A_pos'].values[0][1])
        B_pos = int(df_params.loc[df_params['parameter']=='B_pos'].values[0][1])
        O_pos = int(df_params.loc[df_params['parameter']=='O_pos'].values[0][1])    

    
    vv_list = []
    vv_col = []
    aa_list = []
    aa_col = []

    
    Time = timeNow
    Sigma_v = "{0:.4f}".format(vSum[0] / nMol)

    E = "{0:.4f}".format(totEnergy.sum1)
    Sigma_E = "{0:.4f}".format(totEnergy.sum2)

    Ek = "{0:.4f}".format(kinEnergy.sum1)
    Sigma_Ek = "{0:.4f}".format(kinEnergy.sum2)

    Ep = "{0:.4f}".format(uEnergy.sum1)
    Sigma_Ep = "{0:.4f}".format(uEnergy.sum2)

    P_1 = "{0:.4f}".format(pressure.sum1)
    P_2 = "{0:.4f}".format(pressure.sum2)    
    
    
    #get_ipython().run_line_magic('matplotlib', 'inline')
    #plt.rcParams['axes.facecolor'] = 'white'
    plt.grid(False)
    markersize = 10
    
    #plt.grid(b=None)
    TileName = (workdir+typeOfData+'/'+str(n)+'.png')

    x = []
    y = []
    
    
    for n in range(len(mol)):
        if typeOfData=='coo':
            x.append(mol[n].r[0])
            y.append(mol[n].r[1])
            vv_list.append(mol[n].rv[0] * mol[n].rv[0] + mol[n].rv[1] * mol[n].rv[1])
            
        if typeOfData=='vel':
            x.append(mol[n].rv[0])
            y.append(mol[n].rv[1])
            
        if typeOfData=='acc':
            x.append(mol[n].ra[0])
            y.append(mol[n].ra[1])
            
    # Normalization
    vv_col = (vv_list-min(vv_list))/(max(vv_list)-min(vv_list)) # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range                       
    
    
        
        
    mark_1 = int(len(mol)/2 + len(mol)/8)
    mark_2 = int(len(mol)/2 + len(mol)/8 + 1)
    
    if typeOfData=='coo':

        #plt.plot(x, y, 'o', markeredgecolor='lightgray', markersize=markersize)                
        #plt.plot(x, y, 'o', color='white', markersize=0.1)                

        for j in range(nMol):
            c = np.array([0, vv_col[j], 0, vv_col[j]]) #?

            #plt.plot(x[j], y[j], 'o', color=c)
            #plt.plot(x[j], y[j], 'o', color=c, markersize=1)      
            plt.plot(x[j], y[j], 'o', color=c, markeredgecolor='gray' , markersize=markersize)                         

            if(df_params.loc[df_params['parameter']=='mode'].values[0][1]!='simulation'):

                if j==A_pos:
                    plt.plot(x[j], y[j], 'o', color=c, markeredgecolor='red' , markersize=markersize)
                if j==B_pos:
                    plt.plot(x[j], y[j], 'o', color=c, markeredgecolor='cyan', markersize=markersize)
                if j==O_pos:
                    plt.plot(x[j], y[j], 'o', color=c, markeredgecolor='orange', markersize=markersize)
                

    if typeOfData=='acc':
        plt.plot(x, y, 'o', color='red', markersize=markersize)
    if typeOfData=='vel':
        plt.plot(x, y, 'o', color='green', markersize=markersize)
        
        
    #plt.plot(x[mark_1], y[mark_1], 'o', color='yellow')
    #plt.plot(x[mark_2], y[mark_2], 'o', color='cyan')
    
   
    

    plt.title('timestep:'+"{0:.4f}".format(timeNow)+'; '+'$\Sigma v$:'+Sigma_v+'; '+'E:'+E+'; '+'$\sigma E$:'+Sigma_E+';\n'+'Ek:'+Ek+'; ' +'$\sigma Ek$:'+Sigma_Ek+'; '\
        +'Ep:'+Ep+'; ' +'$\sigma Ep$:'+Sigma_Ep+'; '+'P.sum1:'+P_1+'; '+'P.sum2:'+P_2+'; ', loc='left')
    
    #plt.rcParams["figure.figsize"] = (200,3)
    plt.savefig(TileName, dpi=100)
    plt.clf()
    
    
def makeMov(typeOfData):
    # For more information about the use of the glob package with Python, and for the convertion from 
    # gif to mp4 video formats see:    
    #https://pythonprogramming.altervista.org/png-to-gif/
    #https://stackoverflow.com/questions/6773584/how-is-pythons-glob-glob-ordered
    #https://www.programiz.com/python-programming/datetime/current-time
    #https://stackoverflow.com/questions/40726502/python-convert-gif-to-videomp4
    
    
    t = time.localtime()
    current_time = time.strftime("%D:%H:%M:%S", t)
    current_time = current_time.replace('/','-')


    # Create the frames
    frames = []
    imgs = sorted(glob.glob(typeOfData+'/*.png'), key=os.path.getmtime)
    for i in imgs:
        temp = Image.open(i)
        keep = temp.copy()
        frames.append(keep)
        temp.close()
    #for i in imgs:
    #    os.remove(i)        

    # Save into a GIF file that loops forever
    frames[0].save(typeOfData+'/'+typeOfData+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=30, loop=0)


    clip = mp.VideoFileClip(typeOfData+'/'+typeOfData+'.gif')
    clip.write_videofile(typeOfData+'/'+typeOfData+'_'+current_time+'.mp4')
    #os.remove(typeOfData+'/'+typeOfData+'.gif') # remove gif from disk



def PrintSummary():
    #'''
    print(stepCount, \
          "{0:.4f}".format(timeNow), \
          "{0:.4f}".format(vSum[0] / nMol) ,\
          "{0:.4f}".format(totEnergy.sum1),\
          "{0:.4f}".format(totEnergy.sum2), \
          "{0:.4f}".format(kinEnergy.sum1), \
          "{0:.4f}".format(kinEnergy.sum2),\
          "{0:.4f}".format(uEnergy.sum1), \
          "{0:.4f}".format(uEnergy.sum2),\
          "{0:.4f}".format(pressure.sum1),\
          "{0:.4f}".format(pressure.sum2))
    #'''
    return (stepCount, timeNow, (vSum[0] / nMol) , totEnergy.sum1, totEnergy.sum2, kinEnergy.sum1, kinEnergy.sum2,  uEnergy.sum1, uEnergy.sum2, pressure.sum1,pressure.sum2)    


def GraphOutput():

    ax = df_systemParams.plot(x="timestep", y='$\Sigma v$', kind="line")
    df_systemParams.plot(x="timestep", y='E', kind="line", ax=ax, color="C1")
    df_systemParams.plot(x="timestep", y='$\sigma E$', kind="line", ax=ax, color="C2")
    df_systemParams.plot(x="timestep",  y='Ek', kind="line", ax=ax, color="C3")
    df_systemParams.plot(x="timestep", y='$\sigma Ek$', kind="line", ax=ax, color="C4")
    df_systemParams.plot(x="timestep",  y='Ep', kind="line", ax=ax, color="C5")
    df_systemParams.plot(x="timestep", y='$\sigma Ep$', kind="line", ax=ax, color="C6")
    df_systemParams.plot(x="timestep", y='P_1', kind="line", ax=ax, color="C9")
    df_systemParams.plot(x="timestep", y='P_2', kind="line", ax=ax, color="C9")

    #plt.show()
    plt.savefig('systemParams.jpg', dpi=300)



# HANDLING FUNCTION (SingleStep())
'''
SingleStep: Is the function that handles the processing for a single timestep, including: 
1) the force evaluation
2) integration of the equation of motion, 
3) adjustments required by periodic boundaries, and
4) property measurements
'''
def SingleStep():
    
    global stepCount #  timestep counter
    global timeNow    

    stepCount +=1
    timeNow = stepCount * deltaT
    
    LeapfrogStep(1)
    ApplyBoundaryCond()
    ComputeForces() # 1) The force evaluation
    LeapfrogStep(2) # 2) Integration of coordinates and velocities
    EvalProps()
    AccumProps(1) # Accumulate properties

    if (stepCount % stepAvg == 0):
        AccumProps(2) # Calculate averages
        systemParams.append(PrintSummary())
        #PrintSummary()
        AccumProps(0) # Set to zero all the properties.



def loadLogicPattern(pat):
    # assign pattern values
    A_pos = int(df_params.loc[df_params['parameter']=='A_pos'].values[0][1])
    B_pos = int(df_params.loc[df_params['parameter']=='B_pos'].values[0][1])
    O_pos = int(df_params.loc[df_params['parameter']=='O_pos'].values[0][1])    


    if (fix_coo==1):
        mol[A_pos].r = coo_list[A_pos] 
        mol[B_pos].r = coo_list[B_pos] 
        mol[O_pos].r = coo_list[O_pos] 

    mol[A_pos].rv = np.array([np.float64(df_params.loc[df_params['parameter']=='pattern'].values[pat][1].split(' ')[0]), -np.float64(df_params.loc[df_params['parameter']=='pattern'].values[pat][1].split(' ')[0])])
    mol[B_pos].rv = np.array([np.float64(df_params.loc[df_params['parameter']=='pattern'].values[pat][1].split(' ')[1]), -np.float64(df_params.loc[df_params['parameter']=='pattern'].values[pat][1].split(' ')[1])])
    mol[O_pos].rv = np.array([np.float64(df_params.loc[df_params['parameter']=='pattern'].values[pat][1].split(' ')[2]), -np.float64(df_params.loc[df_params['parameter']=='pattern'].values[pat][1].split(' ')[2])])
    



def loadLogicQuery():
    # assign pattern values
    A_pos = int(df_params.loc[df_params['parameter']=='A_pos'].values[0][1])
    B_pos = int(df_params.loc[df_params['parameter']=='B_pos'].values[0][1])
    O_pos = int(df_params.loc[df_params['parameter']=='O_pos'].values[0][1])        
    

    if (fix_coo==1):
        mol[A_pos].r = coo_list[A_pos] 
        mol[B_pos].r = coo_list[B_pos] 
        mol[O_pos].r = coo_list[O_pos] 



    
    mol[A_pos].rv = np.array([np.float64(df_params.loc[df_params['parameter']=='query'].values[0][1].split(' ')[0]), -np.float64(df_params.loc[df_params['parameter']=='query'].values[0][1].split(' ')[0])])
    mol[B_pos].rv = np.array([np.float64(df_params.loc[df_params['parameter']=='query'].values[0][1].split(' ')[1]), -np.float64(df_params.loc[df_params['parameter']=='query'].values[0][1].split(' ')[1])])

    
    
# MAIN LOOP

#LJ VARIABLES INITIALIZATION

import os.path
from os import path
import shutil

# Set a working directory for all the png and videos
workdir = str(os.getcwd()+'/')

# If the dir /coo doesn't exist make it
if path.exists(str(workdir+'coo'))==False:
    os.makedirs(str(workdir+'coo'))
else:
    shutil.rmtree(str(workdir+'coo'))
    os.makedirs(str(workdir+'coo'))

# If the dir /coo doesn't exist make it
if path.exists(str(workdir+'vel'))==False:
    os.makedirs(str(workdir+'vel'))
else:
    shutil.rmtree(str(workdir+'vel'))
    os.makedirs(str(workdir+'vel'))
    
# If the dir /coo doesn't exist make it
if path.exists(str(workdir+'acc'))==False:
    os.makedirs(str(workdir+'acc'))
else:
    shutil.rmtree(str(workdir+'acc'))
    os.makedirs(str(workdir+'acc'))    


# Load the input parameter file
configuration_file = sys.argv[1]
#configuration_file = 'LJP.in'
df_params = pd.read_csv(configuration_file, sep='\t', header=None, names=['parameter', 'value'])

NDIM = 2 # Two-Dimension setting
vSum = np.asarray([0.0, 0.0]) # velocity sum
kinEnergy =Prop(0.0, 0.0, 0.0) #Ek (and average)
uEnergy =Prop(0.0, 0.0, 0.0) #Ep (and average)
totEnergy =Prop(0.0, 0.0, 0.0) #E (and average)
pressure  =Prop(0.0, 0.0, 0.0) #P (and average) 

systemParams = []

IADD = 453806245
IMUL = 314159269
MASK = 2147483647
SCALE = 0.4656612873e-9
randSeedP = 17

deltaT = float(df_params.values[0][1])
density = float(df_params.values[1][1])

initUcell = np.asarray([0.0, 0.0]) # initialize cell
initUcell[0] = int(df_params.values[2][1])
initUcell[1] = int(df_params.values[3][1])

stepAvg = int(df_params.values[4][1])
stepEquil = float(df_params.values[5][1])
stepLimit = float(df_params.values[6][1])
temperature = float(df_params.values[7][1])
float(df_params.values[7][1])

#Define an array of Mol
mol = [Mol(np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0])) for i in range(int(initUcell[0]*initUcell[1]))]



global nMol
nMol = len(mol)


# LJP parameters:
epsilon =  1
sigma = 1
fix_coo=1 # Fix coordinates for A, B and O

# Oxygen
#epsilon =  0.65
#sigma = 3.1656

coo_list = []


# PARAMETERS
# set mov=1 if you want make a video
mov = df_params.loc[df_params['parameter']=='movie'].values[0][1] 



print('\nSDFL: Soft-Disks Fluid Learning Algorithm')
print('Luca Zammataro\n')


# CLASSIC SIMULATION LOOP

if(df_params.loc[df_params['parameter']=='mode'].values[0][1]=='simulation'):

    SetParams()
    SetupJob()
    moreCycles = 1

    n = 0
    while moreCycles:
        
        SingleStep()
        if mov=='yes':
            plotMol('coo', workdir, n) # Make a graph of the coordinates
        n += 1

        if stepCount >= stepLimit:
            moreCycles = 0
            
        print('SIMULATION: '+sys.argv[1]+' SC:'+str(stepCount)+' SL:'+str(int(stepLimit)))

    columns = ['timestep','timeNow', '$\Sigma v$', 'E', '$\sigma E$', 'Ek', '$\sigma Ek$', 'Ep', '$\sigma Ep$', 'P_1', 'P_2']
    df_systemParams = pd.DataFrame(systemParams, columns=columns) 
    df_systemParams.to_csv('systemParams.csv')           

    # Make a video
    if mov=='yes':
        makeMov('coo')
        #makeMov('acc')
        #makeMov('vel')    

    GraphOutput()




# TRAINING LOOP

if(df_params.loc[df_params['parameter']=='mode'].values[0][1]=='training'):

    SetParams()
    SetupJob()


    n = 1 # number of images
    i = 1


    for i in range(int(df_params.loc[df_params['parameter']=='iterations'].values[0][1])):
        p = 0 # number of patterns

        moreCycles = 1
        while moreCycles:
            moreCycles = 1
            loadLogicPattern(p)
            SingleStep()
            PrintSummary()
            if mov=='yes':
                plotMol('coo', workdir, n) # Make a graph of the coordinates


            if stepCount >= stepLimit:
                stepCount = 0
                p = p + 1
                #if p >= 4:
                if p >= 1:                
                    moreCycles = 0

            print('\tTRAINING: '+sys.argv[1]+' ITE:'+str(i)+' PAT:'+str(p)+' IM:'+str(n)+' SC:'+str(stepCount)+' SL:'+str(int(stepLimit))+\
                ' A:'+str(np.round(mol[int(df_params.loc[df_params['parameter']=='A_pos'].values[0][1])].rv, 2)),\
                ' B:'+str(np.round(mol[int(df_params.loc[df_params['parameter']=='B_pos'].values[0][1])].rv, 2)),\
                ' O:'+str(np.round(mol[int(df_params.loc[df_params['parameter']=='O_pos'].values[0][1])].rv, 2)))
            n += 1

    columns = ['timestep','timeNow', '$\Sigma v$', 'E', '$\sigma E$', 'Ek', '$\sigma Ek$', 'Ep', '$\sigma Ep$', 'P_1', 'P_2']
    df_systemParams = pd.DataFrame(systemParams, columns=columns) 
    df_systemParams.to_csv('systemParams.csv')           

    # Make a video
    if mov=='yes':
        makeMov('coo')

    GraphOutput()

    # Write weights files

    weights_rv = []
    weights_ra = []
    weights_r = []

    for i in range(len(mol)):
        #print(i, mol[i].rv)
        weights_r.append(np.array([mol[i].r[0], mol[i].r[1]]))
        weights_rv.append(np.array([mol[i].rv[0], mol[i].rv[1]]))
        weights_ra.append(np.array([mol[i].ra[0], mol[i].ra[1]]))


    # WRIGHT WEIGHTS    

    df_weights_r = pd.DataFrame(weights_r)
    df_weights_r.to_csv('weights_r.csv')
    df_weights_rv = pd.DataFrame(weights_rv)
    df_weights_rv.to_csv('weights_rv.csv')
    df_weights_ra = pd.DataFrame(weights_ra)
    df_weights_ra.to_csv('weights_ra.csv')





# TESTING LOOP
if(df_params.loc[df_params['parameter']=='mode'].values[0][1]=='testing'):
    
    SetParams()
    SetupJob()
    
    n = 0 # number of images
    A = []
    B = []
    O = []
    O_value_trsh = [] # Output value thresholded
    A_pos = int(df_params.loc[df_params['parameter']=='A_pos'].values[0][1])
    B_pos = int(df_params.loc[df_params['parameter']=='B_pos'].values[0][1])
    O_pos = int(df_params.loc[df_params['parameter']=='O_pos'].values[0][1])


    # Load weights
    df_weights_r = pd.read_csv('weights_r.csv', sep=',')
    df_weights_rv = pd.read_csv('weights_rv.csv', sep=',')
    df_weights_ra = pd.read_csv('weights_ra.csv', sep=',')

    # Retrive weights from the weight file and assing them to mol
    for i in range(nMol):
        mol[i].r = np.array([df_weights_r.values[i][1], df_weights_r.values[i][2]])
        mol[i].rv = np.array([df_weights_rv.values[i][1], df_weights_rv.values[i][2]])
        mol[i].ra = np.array([df_weights_ra.values[i][1], df_weights_ra.values[i][2]]) 


    moreCycles = 1
    while moreCycles:

        # Inject Query into the system with loadLogicQuery
        loadLogicQuery()
        SingleStep()
        if mov=='yes':
            plotMol('coo', workdir, n) # Make a graph of the coordinates
        n += 1
        
        
        if stepCount >= stepLimit:
            moreCycles = 0


        A_value = mol[A_pos].rv[0] * mol[A_pos].rv[0] + mol[A_pos].rv[1] * mol[A_pos].rv[1]
        B_value = mol[B_pos].rv[0] * mol[B_pos].rv[0] + mol[B_pos].rv[1] * mol[B_pos].rv[1]    
        O_value = mol[O_pos].rv[0] * mol[O_pos].rv[0] + mol[O_pos].rv[1] * mol[O_pos].rv[1]    


        if (O_value >= 250.0):
            O_value_trsh.append(1)
        if (O_value < 250.0):
            O_value_trsh.append(0)


        print('TESTING: '+sys.argv[1]+' SC:'+str(stepCount),'SL:'+str(stepLimit), np.round(A_value, 2), np.round(B_value, 2), np.round(O_value, 2))


        A.append(A_value)
        B.append(B_value)
        O.append(O_value)    

    
    columns = ['timestep','timeNow', '$\Sigma v$', 'E', '$\sigma E$', 'Ek', '$\sigma Ek$', 'Ep', '$\sigma Ep$', 'P_1', 'P_2']
    df_systemParams = pd.DataFrame(systemParams, columns=columns) 
    df_systemParams.to_csv('systemParams.csv')           


    # Make a video
    if mov=='yes':
        makeMov('coo')

    GraphOutput()



    # MAKE OUTPUTS

    myData = [A, B, O]
    df_results = pd.DataFrame(myData).T
    df_results = df_results.set_axis(['A', 'B', 'O'], axis=1)

    #plot_name = 'A='+str(np.round(A_value,2))+'_'+    'B='+str(np.round(B_value,2))+'_'+    'AVE='+str(np.round(df_results['O'].mean(), 2))+'_'+    'ST='+str(np.round(df_results['O'].std(), 2))+'_'+    'MIN='+str(np.round(df_results['O'].min(),2))+'_'+    'MAX='+str(np.round(df_results['O'].max(),2))+'_'+    'cycl='+str(stepLimit)
    plot_name = sys.argv[1][23:-4]

    fig = df_results.plot(figsize=(6, 4), fontsize=10).get_figure()
    fig.savefig(plot_name+'.rawdata'+'.pdf')

    # Plot the Output pick (threshold >= 250)
    df_O_trsh = pd.DataFrame(O_value_trsh)
    df_O_trsh = df_O_trsh.set_axis(['O'], axis=1)
    fig = df_O_trsh.plot(figsize=(6, 4), fontsize=10).get_figure()
    #fig.savefig(plot_name+'.threshold'+'.pdf')
    df_results.to_csv(plot_name+'.rawdata'+'.csv')






