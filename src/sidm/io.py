import h5py
import numpy as np

class HaloReader:
    """
    A class to read halo data from HDF5 files.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self._load_data()

    def _load_data(self):
        """
        Loads data from the HDF5 file.
        """
        with h5py.File(self.file_path, "r") as f:
            # Populate attributes for easier access
            def load_group(group_dict):
                for key, value in group_dict.items():
                    if isinstance(value, h5py.Dataset):
                        setattr(self, key, value[()]) # Load into memory
                    else:
                        setattr(self, key, value)

            if "Halo_data" in f:
                load_group(dict(f["Halo_data"].items()))
            
            if "Profile_data" in f:
                load_group(dict(f["Profile_data"].items()))
    
    def get_subset(self, condition):
        """
        Returns indices satisfying a condition.
        Example: halo.get_subset(lambda h: h.kappa > 0.3)
        """
        # This requires known structure, assuming arrays in attributes
        # For now, we leave the filtering to the user or implement specific filters as methods
        pass

def calculate_mass(density_profile, radius_bins, Reff):
    """
    Calculates the integrated mass up to a radius Reff.
    density_profile : Density profile (array).
    radius_bins : Corresponding radii (array).
    Reff : Effective radius to integrate up to (float).
    """
    valid_indices = radius_bins <= Reff
    r = radius_bins[valid_indices]
    rho = density_profile[valid_indices]

    if len(r) == 0:
        return 0.0

    # Discrete integration (similar to trapz)
    mass = np.trapz(4 * np.pi * r**2 * rho, r)
    return mass
