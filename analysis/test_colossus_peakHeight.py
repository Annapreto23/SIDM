from colossus.cosmology import cosmology
from colossus.lss import peaks
import numpy as np

params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.96}
cosmology.addCosmology('test', params)
c = cosmology.setCosmology('test')

M_arr = np.logspace(10, 15, 5)
try:
    nu = peaks.peakHeight(M_arr, 0.0)
    print(f"Output type: {type(nu)}")
    print(f"Output: {nu}")
except Exception as e:
    print(f"Error: {e}")
