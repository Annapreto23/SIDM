import h5py
import numpy as np
import os

# Create data directory if not exists
if not os.path.exists("data"):
    os.makedirs("data")

file_path = "data/dummy_halo.hdf5"
if not os.path.exists(file_path):
    with h5py.File(file_path, "w") as f:
        g_halo = f.create_group("Halo_data")
        g_profile = f.create_group("Profile_data")

        # Dummy data matching keys used in scripts
        g_halo.create_dataset("kappa", data=np.random.rand(10))
        g_halo.create_dataset("AxisRadius", data=np.linspace(0.1, 10, 100))
        g_halo.create_dataset("GalaxyProjectedHalfLightRadius", data=np.random.rand(10))
        g_halo.create_dataset("M200c", data=np.random.rand(10)*10 + 10) # Log mass
        g_halo.create_dataset("Mstar", data=np.random.rand(10)*10 + 9) # Log mass
        g_halo.create_dataset("c200c", data=np.random.rand(10)*10)
        
        # Profile data
        g_profile.create_dataset("Dark_matter_Density_profile", data=np.random.rand(100, 10))
        g_profile.create_dataset("Density_profile", data=np.random.rand(100, 10))
        g_profile.create_dataset("Density_radial_bins", data=np.linspace(0.1, 10, 100))
        g_profile.create_dataset("Dark_matter_Sigma_profile", data=np.random.rand(100, 10))
