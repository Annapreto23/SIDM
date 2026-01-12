
#---user modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import sidm.profiles as pr
import sidm.galhalo as gh
from sidm.io import HaloReader as classhalo # Use HaloReader as drop-in replacement
from sidm.io import calculate_mass # If needed

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


#################
# Plot parameters
params = {
    "font.size": 20,
    "font.family": "Arial Black",
    "text.usetex": True,
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

# class classhalo removed, using HaloReader from sidm.io

file_paths = ['/home/ananas/Forks/TangoSIDM_DMFraction/AnalyticalModelBis/Halo_data_L025N376WeakStellarFBSigmaVelDep60Anisotropic_withgas.hdf5',
              '/home/ananas/Forks/TangoSIDM_DMFraction/data/Simulation_datasets/TangoSIDM/Halo_data_L025N376ReferenceSigmaVelDep30Anisotropic.hdf5',
              '/home/ananas/Forks/TangoSIDM_DMFraction/data/Simulation_datasets/TangoSIDM/Halo_data_L025N376ReferenceSigmaConstant00.hdf5', 
              'SigmaConstant10.hdf5']

file_names = ['WSFB60', 'WSFB30', 'WSFBconst', 'Ref60', 'Ref30', 'Refconst']

# Ensure dummy data usage or user provided path if original paths are invalid
# For now, we keep original paths as comments or try-catch, or just let it fail if file missing (user needs to configure)
# But since I created dummy data, I might point to it for testing
# halo = classhalo("data/dummy_halo.hdf5") 
# But let's respect the original script's intent, maybe adding a comment.

try:
    halo = classhalo(file_paths[1])
except:
    print(f"File {file_paths[1]} not found, falling back to dummy data if available or failing.")
    if os.path.exists("../data/dummy_halo.hdf5"):
        halo = classhalo("../data/dummy_halo.hdf5")
    else:
        print("Dummy data not found either. Please configure file_paths.")
        sys.exit(1)


# The rest of the script logic...
# I need to copy the logic from original phase.py but with adapted imports.
# Since write_to_file overwrites, I must include the logic.
# I will copy the logic from Step 20 from line 128 (calculate_mass is in io.py, so I skip it or import it).
# Logic starts at line 143: sig = []...

sig = []
true_sig = []
# Ensure we don't go out of bounds if dummy data is small
loop_range = min(20, len(halo.Mstar)) if hasattr(halo, 'Mstar') else 0

for k in range(loop_range):
    
    #---target CDM halo and baryon distribution
    Reff = halo.GalaxyProjectedHalfLightRadius[k]
    lgMv = halo.M200c[k] # [M_sun]
    c = halo.c200c[k]
    lgMb = halo.Mstar[k] # [M_sun]
    r0 = Reff / (1 + np.sqrt(2)) # [kpc]

    r_FullRange = np.logspace(-3,3,500) # [kpc] for plotting the full profile

    ############################### compute #################################

    print('>>> computing SIDM profile ... ')

    #---prepare the CDM profile to stitch to
    # with baryons
    Mv = 10.**lgMv
    Mb = 10.**lgMb
    halo_init = pr.NFW(Mv,c,Delta=100.,z=0.)
    disk = pr.Hernquist(Mb,r0)
    halo_contra = gh.contra(r_FullRange,halo_init,disk)[0] # <<< adiabatically contracted CDM halo
    tage = 9

    sigmamx = np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.Density_radial_bins, halo.Dark_matter_Sigma_profile[:, k])
    radius = np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.Density_radial_bins, np.arange(len(halo.Density_radial_bins)))
    print(sigmamx)
    print('len de sig, ',radius)
    sigmamx_halo = sigmamx
    
    # dark-matter only
    fb = Mb/Mv
    disk_dmo = pr.Hernquist(0.001,100.) # <<< use a tiny mass and huge size for the DM-only case
    halo_dmo = pr.NFW((1.-fb)*Mv,c,Delta=halo_init.Deltah,z=halo_init.z) # <<< DMO CDM halo

    #---find r_1

    #---with baryon

    sigmamx_values = np.linspace(-2, 10, 30)
    sigmamx_theo = []
    fraction_dm_map = []
    for j, sigmamx in enumerate(sigmamx_values):
        try:
            r1 = pr.r1(halo_contra,sigmamx=sigmamx,tage=tage)
            r1_dmo = pr.r1(halo_dmo,sigmamx=sigmamx,tage=tage)
        except ValueError:
            continue
        rhodm0,sigma0,rho,Vc,r = pr.stitchSIDMcore(r1,halo_contra,disk)

    #---dark-matter only
        rhodm0_dmo,sigma0_dmo,rho_dmo,Vc_dmo,r_dmo = pr.stitchSIDMcore(r1_dmo,halo_dmo,disk_dmo)
        sigmamx_theo.append(np.interp(r1, halo.Density_radial_bins, halo.Dark_matter_Sigma_profile[:, k]))
        # Calcul de la fraction de matière noire
        M_dm_Reff = calculate_mass(rho, r, Reff)
        M_total_Reff = calculate_mass(halo.Density_profile[:, k], halo.Density_radial_bins, Reff)
        fraction_dm_map.append(M_dm_Reff / M_total_Reff if M_total_Reff > 0 else 0)
    
    # Handling potential missing fDM if input data is different
    if hasattr(halo, 'fDM'):
        f = np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.AxisRadius, halo.fDM[:, k])
        print('f is ', f)
        print(fraction_dm_map)
        
        if len(fraction_dm_map) > 0:
            index_closest = np.argmin(np.abs(np.array(fraction_dm_map) - f))  # Indice de la valeur la plus proche
            print(np.array(fraction_dm_map) - f)
            sigmamx_closest = sigmamx_values[index_closest] 
            sig.append(sigmamx_closest)
            true_sig.append(sigmamx_theo[index_closest])

print(sig)
print(true_sig)
plt.figure()
plt.scatter(sig, true_sig)
# plt.show() # Commented out for batch execution
plt.savefig("phase_space_scatter.png")