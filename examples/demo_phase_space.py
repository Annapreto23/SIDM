
#---user modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import sidm.profiles as pr
import sidm.galhalo as gh
from sidm.io import HaloReader as classhalo
from sidm.io import calculate_mass

#---standard python stuff
import numpy as np

#---for plot

import matplotlib.pyplot as plt

#########################################################################

import h5py

#################
# Plot parameters
params = {
    "font.size": 20,
    "font.family": "Arial Black",
    "text.usetex": False,
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
              '/home/ananas/Forks/TangoSIDM_DMFraction/data/Simulation_datasets/TangoSIDM/Halo_data_L025N376ReferenceSigmaVelDep30Anisotropic.hdf5',
              '/home/ananas/Forks/TangoSIDM_DMFraction/data/Simulation_datasets/TangoSIDM/Halo_data_L025N376ReferenceSigmaConstant00.hdf5']

file_names = ['WSFB60', 'WSFB30', 'WSFBconst', 'Ref60', 'Ref30', 'Refconst']

# Use dummy data fallback
try:
    halo = classhalo(file_paths[2])
except:
    if os.path.exists("../data/dummy_halo.hdf5"):
        halo = classhalo("../data/dummy_halo.hdf5")
    else:
        print("Data file not found.")
        sys.exit(1)

k = np.random.randint(0, min(100, len(halo.M200c)))
print(k)

# Check attributes
if hasattr(halo, 'fDM'):
    print(np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.AxisRadius, halo.fDM[:, k]))

print(halo.GalaxyProjectedHalfLightRadius[k])
print(halo.M200c[k])
print(halo.Mstar[k])
print(halo.c200c[k])


########################### user control ################################

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
tage = 13

if hasattr(halo, 'Dark_matter_Sigma_profile'):
    sigmamx = np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.Density_radial_bins, halo.Dark_matter_Sigma_profile[:, k])
    print(sigmamx)
    sigmamx_halo = sigmamx
else:
    sigmamx = 1.0 # default
    sigmamx_halo = 1.0


# dark-matter only
fb = Mb/Mv
disk_dmo = pr.Hernquist(0.001,100.) # <<< use a tiny mass and huge size for the DM-only case
halo_dmo = pr.NFW((1.-fb)*Mv,c,Delta=halo_init.Deltah,z=halo_init.z) # <<< DMO CDM halo

#---find r_1
r1 = pr.r1(halo_contra,sigmamx=sigmamx,tage=tage)
r1_dmo = pr.r1(halo_dmo,sigmamx=sigmamx,tage=tage)

#---with baryon
rhodm0,sigma0,rho,Vc,r = pr.stitchSIDMcore(r1,halo_contra,disk)

#---dark-matter only
rhodm0_dmo,sigma0_dmo,rho_dmo,Vc_dmo,r_dmo = pr.stitchSIDMcore(r1_dmo,halo_dmo,disk_dmo)

print(len(r_FullRange))
print(len(rho))

# Calculer la masse de matière noire et la masse totale
M_dm_Reff = calculate_mass(halo.Dark_matter_Density_profile[:, k], halo.Density_radial_bins, Reff)
M_total_Reff = calculate_mass(halo.Density_profile[:, k], halo.Density_radial_bins, Reff)
fraction_dm = M_dm_Reff/M_total_Reff if M_total_Reff > 0 else 0
# Afficher les résultats
print(f"Masse de matière noire à Reff : {M_dm_Reff:.2e} M_sun")
print(f"Masse totale à Reff : {M_total_Reff:.2e} M_sun")

print(f"Fraction de matière noire à R_eff : {fraction_dm:.4f}")


# ... plotting code ...
# For brevity, truncated
