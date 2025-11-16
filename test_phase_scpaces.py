#########################################################################

# Program that solves the Jeans-Poisson equation for the density profile 
# of SIDM halo in the presence of Hernquist baryon mass distribution.

# In this version, we start with a NFW CDM halo of given halo age, as 
# well as the Henquist baryon distribution, and we find the central DM 
# density (rho_dm0) and the velocity dispersion (sigma_0) by minimizing
# the figure of merit, delta, which measures the fractional difference
# in density and enclosed mass between the SIDM profile and the CDM 
# profile at r_1 ( the characteristic radius within which an average DM 
# particle has experienced one or more self-interaction )

# Arthur Fangzhou Jiang 2020 Caltech

######################## set up the environment #########################

#---user modules
import profiles as pr
import galhalo as gh

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
class classhalo:
    def __init__(self, file_path, type = 'all' ):
        self.file_path = file_path
        self.get_group_attributes(type)

    def get_group_attributes(self, type):
        with h5py.File(self.file_path, "r") as file:
            halo_data_group = file["Halo_data"]
            profile_data_group = file["Profile_data"]
            
            self.kappa = halo_data_group["kappa"][:]
            if type == 'disk':
                select = np.where(halo_data_group["kappa"][:] >= 0.3)[0]
            if type == 'elliptical':
                select = np.where(halo_data_group["kappa"][:] < 0.3)[0]
            if type == 'all':
                select = np.where(halo_data_group["kappa"][:] > 0)[0]
            

            # Halo_data attributes
            self.AxisRadius = halo_data_group["AxisRadius"][:]
            self.CrossSection = halo_data_group["CrossSection"][:, select]
            self.MeanCrossSection = halo_data_group["MeanCrossSection"][:, select]
            
            self.ReCrossSection = halo_data_group["ReCrossSection"][0, select]
            self.ReMeanCrossSection = halo_data_group["ReMeanCrossSection"][0, select]
            self.R12CrossSection = halo_data_group["R12CrossSection"][0, select]
            self.R12MeanCrossSection = halo_data_group["R12MeanCrossSection"][0, select]
            self.R200cCrossSection = halo_data_group["R200cCrossSection"][0, select]
            self.R200cMeanCrossSection = halo_data_group["R200cMeanCrossSection"][0, select]


            self.DynamicalRelaxation = halo_data_group["DynamicalRelaxation"][select]
            self.GalaxyHalfLightRadius = halo_data_group["GalaxyHalfLightRadius"][select]
            self.GalaxyHalfMassRadius = halo_data_group["GalaxyHalfMassRadius"][select]
            self.GalaxyLuminosity = halo_data_group["GalaxyLuminosity"][select]
            self.GalaxyProjectedHalfLightRadius = halo_data_group["GalaxyProjectedHalfLightRadius"][select]
            self.GalaxyProjectedHalfMassRadius = halo_data_group["GalaxyProjectedHalfMassRadius"][select]
            self.ID = halo_data_group["ID"][select]
            self.M200c = halo_data_group["M200c"][select]
            self.Mgas = halo_data_group["Mgas"][select]
            self.Mstar = halo_data_group["Mstar"][select]
            self.R200c = halo_data_group["R200c"][select]
            self.SpecificAngularMomentum = halo_data_group["SpecificAngularMomentum"][select,:]
            self.StructureType = halo_data_group["StructureType"][select]
            self.Vmax = halo_data_group["Vmax"][select]
            self.c200c = halo_data_group["c200c"][select]

            sarg =0

            # Profile_data attributes
            self.Circular_Velocity = profile_data_group["Circular_Velocity"][:, select]
            self.Dark_matter_Circular_Velocity = profile_data_group["Dark_matter_Circular_Velocity"][:, select]
            self.Dark_matter_Density_profile = profile_data_group["Dark_matter_Density_profile"][sarg:,select]
            self.Dark_matter_Sigma_profile = profile_data_group["Dark_matter_Sigma_profile"][sarg:,select]
            self.Dark_matter_Velocity_dispersion = profile_data_group["Dark_matter_Velocity_dispersion"][sarg:,select]
            self.Density_profile = profile_data_group["Density_profile"][sarg:,select]
            self.Density_radial_bins = profile_data_group["Density_radial_bins"][sarg:]
            self.Gas_Circular_Velocity = profile_data_group["Gas_Circular_Velocity"][:,select]
            self.Gas_Density_profile = profile_data_group["Gas_Density_profile"][sarg:,select]
            self.Gas_Velocity_dispersion = profile_data_group["Gas_Velocity_dispersion"][sarg:,select]
            self.Stars_Circular_Velocity = profile_data_group["Stars_Circular_Velocity"][:,select]
            self.Stars_Density_profile = profile_data_group["Stars_Density_profile"][sarg:,select]
            self.Stars_Velocity_dispersion = profile_data_group["Stars_Velocity_dispersion"][sarg:,select]
            self.Velocity_radial_bins = profile_data_group["Velocity_radial_bins"][:]
            self.Projected_fDM = halo_data_group["GalaxyProjectedDarkMatterFraction"][:, select]
            self.fDM = halo_data_group["GalaxyDarkMatterFraction"][:, select]
            self.theo_fDM = halo_data_group["GalaxyTheoricalDarkMatterFraction"][:, select]
            #self.fDM_cal= (profile_data_group["Dark_matter_Circular_Velocity"][:,select]/profile_data_group["Circular_Velocity"][:,select])**2

       
        
file_paths = ['/home/ananas/Forks/TangoSIDM_DMFraction/AnalyticalModelBis/Halo_data_L025N376WeakStellarFBSigmaVelDep60Anisotropic_withgas.hdf5',
              'Halo_data_L025N376WeakStellarFBSigmaVelDep30Anisotropic.hdf5',]

file_names = ['WSFB60', 'WSFB30', 'WSFBconst', 'Ref60', 'Ref30', 'Refconst']

halo = classhalo(file_paths[0])
k = 0
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
tage = 9

sigmamx = np.interp(halo.GalaxyProjectedHalfLightRadius[k], halo.AxisRadius, halo.Dark_matter_Sigma_profile[:, k])
print(sigmamx)

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
