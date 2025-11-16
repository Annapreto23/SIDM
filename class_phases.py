import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.integrate import simps
import profiles as pr
import galhalo as gh

#---standard python stuff
import numpy as np

#---for plot

import matplotlib.pyplot as plt

from scipy.integrate import fixed_quad


def calculate_mass(density_profile, radius_bins, Reff):
    """
    Calcule la masse intégrée jusqu'à un rayon Reff.
    density_profile : Profil de densité (array).
    radius_bins : Rayons correspondants (array).
    Reff : Rayon effectif jusqu'où intégrer (float).
    """
    # Filtrer les rayons jusqu'à Reff
    valid_indices = radius_bins <= Reff
    r = radius_bins[valid_indices]
    rho = density_profile[valid_indices]

    # Intégration discrète (similaire à trapz)
    mass = np.trapz(4 * np.pi * r**2 * rho, r)
    return mass

class HaloAnalysis:
    def __init__(self, file_path):
        """
        Initialise l'objet HaloAnalysis avec les données HDF5.
        """
        self.file_path = file_path
        self._load_data()

    def _load_data(self):
        """
        Charge les données du fichier HDF5 et initialise les attributs nécessaires.
        """
        with h5py.File(self.file_path, "r") as file:
            self.halo_data = file["Halo_data"]
            self.profile_data = file["Profile_data"]

            # Charger les données nécessaires
            self.kappa = self.halo_data["kappa"][:]
            self.AxisRadius = self.halo_data["AxisRadius"][:]
            self.GalaxyProjectedHalfLightRadius = self.halo_data["GalaxyProjectedHalfLightRadius"][:]
            self.M200c = self.halo_data["M200c"][:]
            self.Mstar = self.halo_data["Mstar"][:]
            self.c200c = self.halo_data["c200c"][:]
            self.DarkMatterDensityProfile = self.profile_data["Dark_matter_Density_profile"][:]
            self.DensityProfile = self.profile_data["Density_profile"][:]
            self.DensityRadialBins = self.profile_data["Density_radial_bins"][:]
            self.Dark_matter_Sigma_profile = self.profile_data["Dark_matter_Sigma_profile"][:]

    def get_properties(self, index):
        """
        Retourne les propriétés pour un index donné.
        """
        M_dm_Reff = calculate_mass(self.DarkMatterDensityProfile[:, index], self.DensityRadialBins, self.GalaxyProjectedHalfLightRadius[index])
        M_total_Reff = calculate_mass(self.DensityProfile[:, index], self.DensityRadialBins, self.GalaxyProjectedHalfLightRadius[index])
        fraction_dm = M_dm_Reff/M_total_Reff
        properties = {
            "ProjectedHalfLightRadius": self.GalaxyProjectedHalfLightRadius[index],
            "M200c": self.M200c[index],
            "Mstar": self.Mstar[index],
            "c200c": self.c200c[index],
            "sigma": self.Dark_matter_Sigma_profile[:, index],
            "density": self.DensityProfile[:, index], 
            "fDM": fraction_dm
        }
        return properties

    def run(self):
        crosssection = []
        errors = []
        true_crosssection = []
        for i in range(len(self.Mstar)):
            halo = self.get_properties(i)
            Reff = halo["ProjectedHalfLightRadius"]
            lgMv = halo["M200c"] # [M_sun]
            c = halo["c200c"]
            lgMb = halo["Mstar"] # [M_sun]
            sigma = halo["sigma"]
            density_profile = halo["density"]
            r0 = Reff / (1 + np.sqrt(2)) 
            r_FullRange = np.logspace(-3,3,500)
            fDM = halo["fDM"]
            sig, delta = self.calculate_sigma(lgMb, lgMv, c, r0, r_FullRange, Reff, density_profile, fDM)
            crosssection.append(sig)
            true_crosssection.append(sigma)
            errors.append(delta)
        return crosssection, true_crosssection, errors

    def calculate_sigma(self, lgMb, lgMv, c, r0, r_FullRange, Reff, density_profile, fDM):
        Mv = 10.**lgMv
        Mb = 10.**lgMb
        halo_init = pr.NFW(Mv,c,Delta=100.,z=0.)
        disk = pr.Hernquist(Mb,r0)
        halo_contra = gh.contra(r_FullRange,halo_init,disk)[0] # <<< adiabatically contracted CDM halo
        tage = 13
        sigmamx_values = np.linspace(-2, 10, 10)  # Plage de sigma/mx
        fraction_dm_map = np.zeros(len(sigmamx_values))  # Matrice pour stocker les fractions DM


        for j, sigmamx in enumerate(sigmamx_values):
            try:
                r1 = pr.r1(halo_contra,sigmamx=sigmamx,tage=tage)
                rhodm0, sigma0, rho, Vc, r = pr.stitchSIDMcore(r1, halo_contra, disk)
                    # Calcul de la fraction de matière noire
                M_dm_Reff = calculate_mass(rho, r, Reff)
                M_total_Reff = calculate_mass(density_profile, self.DensityRadialBins, Reff)
                fraction_dm_map[j] = M_dm_Reff / M_total_Reff
            except ValueError:
                fraction_dm_map[j] = 0
                continue
            # Calcul du profil avec baryons

        index_closest = np.argmin(np.abs(fraction_dm_map - fDM))
        sigma_adjusted =  sigmamx_values[index_closest] 
        delta = (abs(fDM - fraction_dm_map[index_closest])/ fDM)*sigma_adjusted
        return sigma_adjusted, delta
        

file_path = '/home/ananas/Forks/TangoSIDM_DMFraction/data/Simulation_datasets/TangoSIDM/Halo_data_L025N376ReferenceSigmaVelDep30Anisotropic.hdf5'
halo_analysis = HaloAnalysis(file_path)
crosssection, true_crosssection, errors = halo_analysis.run()

with open("results.txt", "w") as f:
    f.write(str(crosssection))
    f.write(str(errors))

plt.figure()

plt.errorbar(crosssection, true_crosssection, yerr = errors, marker = 'o')




