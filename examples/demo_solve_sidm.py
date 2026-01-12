#########################################################################

# Program that solves the Jeans-Poisson equation for the density profile 
# of SIDM halo in the presence of Hernquist baryon mass distribution.

######################## set up the environment #########################

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

#---user modules
import sidm.profiles as pr
import sidm.galhalo as gh
import sidm.config as cfg # Ensure config is imported if needed implicitly
#---standard python stuff
import numpy as np

#---for plot

import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad

#########################################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob
from scipy import stats
from sidm.io import HaloReader as classhalo

#################
# Plot parameters
params = {
    "font.size": 20,
    "font.family": "Arial Black",
    "text.usetex": False, # Changed to False to avoid requirement for LaTeX
    "mathtext.fontset": "custom",
    "figure.figsize": (4, 3),
    "figure.subplot.left": 0.15,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.16,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": 0.3,
    "figure.subplot.hspace": 0.3,
    "lines.markersize": 2,
    "lines.linewidth": 1.5,
}
plt.rcParams.update(params)
G = 6.67e-11
K = 1.65**2
kpc = 3.085e16*1e3

file_paths = ['/home/ananas/Forks/TangoSIDM_DMFraction/AnalyticalModelBis/Halo_data_L025N376WeakStellarFBSigmaVelDep60Anisotropic_withgas.hdf5',
              'Halo_data_L025N376WeakStellarFBSigmaVelDep30Anisotropic.hdf5',]

file_names = ['WSFB60', 'WSFB30', 'WSFBconst', 'Ref60', 'Ref30', 'Refconst']

# Fallback to dummy data
try:
    halo = classhalo(file_paths[0])
except:
     if os.path.exists("../data/dummy_halo.hdf5"):
        halo = classhalo("../data/dummy_halo.hdf5")
     else:
        print("Data file not found. Please configure file_paths.")
        sys.exit(1)

k = 0
# Ensure k is valid
if k >= len(halo.M200c):
    k=0

# Check attributes existence before usage
if hasattr(halo, 'fDM'):
    print(np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.AxisRadius, halo.fDM[:, k]))

print(halo.GalaxyProjectedHalfLightRadius[k])
print(halo.M200c[k])
print(halo.Mstar[k])
print(halo.c200c[k])

########################### user control ################################

#---target CDM halo and baryon distribution

lgMv = halo.M200c[k] # [M_sun]
c = halo.c200c[k]
lgMb = halo.Mstar[k] # [M_sun]
r0 = halo.GalaxyProjectedHalfLightRadius[k] / (1 + np.sqrt(2)) # [kpc]
    
r_FullRange = np.logspace(-3,3,200) # [kpc] for plotting the full profile

############################### compute #################################

print('>>> computing SIDM profile ... ')

#---prepare the CDM profile to stitch to
# with baryons
Mv = 10.**lgMv
Mb = 10.**lgMb
halo_init = pr.NFW(Mv,c,Delta=100.,z=0.)
disk = pr.Hernquist(Mb,r0)
halo_contra = gh.contra(r_FullRange,halo_init,disk)[0] # <<< adiabatically contracted CDM halo

def calculate_derivative(density, r):
    """Calculate the numerical derivative of the density profile."""
    return np.gradient(density, r)

def calculate_distance(deriv1, deriv2):
    """Calculate the distance between two derivative profiles."""
    return np.mean((np.abs(deriv1 - deriv2)) / np.abs(deriv1) * 100)

def dens(r, density, radius):
    return np.interp(r, radius, density)

mass = lambda r, density, radius: 4*np.pi*dens(r, density, radius)*r**2

def find_best_match(density1, density2, r):
    
    n = len(density1)
    best_r = []
    thresold = 3
    best_distance = []

    for i in range(0, n - thresold):
        deriv1 = calculate_derivative(density1[i:i+thresold], r[i:i+thresold])
        deriv2 = calculate_derivative(density2[i:i+thresold], r[i:i+thresold])
        dist = calculate_distance(deriv1, deriv2)
        M1 = fixed_quad(mass, 0, r[i], args=(density1, r))[0]
        M2 = fixed_quad(mass, 0, r[i], args=(density2, r))[0]
        # print("masses are ", M1/1e8, " and ",M2/1e8)
        distM = calculate_distance(M1, M2)
        # print("distM is ", distM)
        if dist < 25 and distM < 25:
            # print("Condition completed")
            return r[i], dist
        else:
            best_distance.append(dist)
            best_r.append(i)

    mini = np.argmin(best_distance)

    return r[best_r[mini]], best_distance[mini]


r1, dist = find_best_match(halo_contra.rho(r_FullRange), halo_init.rho(r_FullRange), r_FullRange)


print(f"r1 is :  {r1:.2f} kpc")
print(f"Dist is :  {dist}")


rhor1 = np.interp(r1, r_FullRange, halo_contra.rho(r_FullRange))
print(rhor1)

# Visualisation
plt.plot(r_FullRange, halo_contra.rho(r_FullRange), label='rho iso')
plt.plot(r_FullRange, halo_init.rho(r_FullRange), label='CDM')
plt.axvline(x=r1, color='r', linestyle='--', label=f'Optimal r = {r1}')
plt.xlabel('Rayon')
plt.ylabel('Densité')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.savefig("demo_solve_sidm.png")

#######################################
kpctocm = 3.086e16*1e5
kmtocm = 1e5
vel_disp = 200 #km/s
vel_disp *= kmtocm #cm/s
tage = 10*1e9 #m
tage *= 60*60*24*361
rho = rhor1 #Msol/kpc3
rho *= 2e30*1e3
rho /= kpctocm**3

def sigm(rho, vel, tage):
    return np.pi/(rho*vel*4*tage)

print("sigm", sigm(rho, vel_disp, tage))
# print(np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.Density_radial_bins, halo.Dark_matter_Sigma_profile[:, k]))
