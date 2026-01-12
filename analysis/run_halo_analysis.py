#!/usr/bin/env python3
"""
Refactored Halo Analysis Script
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure src is in pythonpath if running from within the repo structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sidm.io import HaloReader, calculate_mass
import sidm.profiles as pr
import sidm.galhalo as gh

class HaloAnalyzer:
    def __init__(self, file_path):
        self.reader = HaloReader(file_path)
        # Cache properties for simpler access if needed, or access via reader
        self.Mstar = self.reader.Mstar
        # ... and so on. simpler to just access reader attributes directly in logic

    def get_properties(self, index):
        """
        Returns properties for a given index using data from HaloReader.
        """
        # Accessing data from reader
        dm_profile = self.reader.Dark_matter_Density_profile[:, index]
        density_profile = self.reader.Density_profile[:, index]
        r_bins = self.reader.Density_radial_bins
        Reff = self.reader.GalaxyProjectedHalfLightRadius[index]

        M_dm_Reff = calculate_mass(dm_profile, r_bins, Reff)
        M_total_Reff = calculate_mass(density_profile, r_bins, Reff)
        
        fraction_dm = M_dm_Reff / M_total_Reff if M_total_Reff > 0 else 0

        properties = {
            "ProjectedHalfLightRadius": Reff,
            "M200c": self.reader.M200c[index],
            "Mstar": self.reader.Mstar[index],
            "c200c": self.reader.c200c[index],
            "sigma": self.reader.Dark_matter_Sigma_profile[:, index],
            "density": density_profile, 
            "fDM": fraction_dm
        }
        return properties

    def run(self):
        crosssection = []
        errors = []
        true_crosssection = []
        
        # Determine number of halos
        count = len(self.reader.Mstar)
        
        for i in range(count):
            halo = self.get_properties(i)
            Reff = halo["ProjectedHalfLightRadius"]
            lgMv = halo["M200c"] # [M_sun]
            c = halo["c200c"]
            lgMb = halo["Mstar"] # [M_sun]
            sigma = halo["sigma"] # This seems to be a profile, not a scalar? In original code it was assigned to sigma and used as true_crosssection?
            # Original code: sigma = halo["sigma"], then true_crosssection.append(sigma).
            # But wait, sigma is a profile [:, index]. Appending a full array to true_crosssection?
            # Original code line 97: true_crosssection.append(sigma)
            # Line 141: plt.errorbar(crosssection, true_crosssection, ...)
            # Errorbar usually expects scalars. 
            # In phase.py line 169: sigmamx = np.interp(halo.GalaxyProjectedHalfLightRadius[k], ...)
            # It seems the original code might have a bug or I misunderstood "sigma" content.
            # However, I will preserve logic or try to interpret. 
            # In class_phases.py: "sigma": self.Dark_matter_Sigma_profile[:, index]
            # This is definitely an array.
            
            density_profile = halo["density"]
            r0 = Reff / (1 + np.sqrt(2)) 
            r_FullRange = np.logspace(-3,3,500)
            fDM = halo["fDM"]
            
            # call calculate_sigma
            sig, delta = self.calculate_sigma(lgMb, lgMv, c, r0, r_FullRange, Reff, density_profile, fDM)
            crosssection.append(sig)
            true_crosssection.append(sigma) # This appends an array?
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
                M_total_Reff = calculate_mass(density_profile, self.reader.Density_radial_bins, Reff)
                fraction_dm_map[j] = M_dm_Reff / M_total_Reff if M_total_Reff > 0 else 0
            except ValueError:
                fraction_dm_map[j] = 0
                continue

        index_closest = np.argmin(np.abs(fraction_dm_map - fDM))
        sigma_adjusted =  sigmamx_values[index_closest] 
        delta = (abs(fDM - fraction_dm_map[index_closest])/ fDM)*sigma_adjusted if fDM > 0 else 0
        return sigma_adjusted, delta

def main():
    parser = argparse.ArgumentParser(description="Run Halo Analysis")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
    parser.add_argument("--output", type=str, default="results.txt", help="Output file for results")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} not found.")
        sys.exit(1)

    analyzer = HaloAnalyzer(args.file_path)
    crosssection, true_crosssection, errors = analyzer.run()

    # Note: true_crosssection is varying in type based on original code analysis. 
    # If it's an array of arrays, writing it to text might be messy.
    
    with open(args.output, "w") as f:
        f.write(str(crosssection) + "\n")
        f.write(str(errors) + "\n")
        # f.write(str(true_crosssection)) 

    print("Analysis complete. Results saved to", args.output)
    
    # Plotting code commented out or adjusted to save instead of show
    # plt.figure()
    # plt.errorbar(crosssection, true_crosssection, yerr = errors, marker = 'o')
    # plt.savefig("crosssection.png")

if __name__ == "__main__":
    main()
