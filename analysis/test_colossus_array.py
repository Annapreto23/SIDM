from colossus.cosmology import cosmology
import numpy as np

params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.96}
cosmology.addCosmology('test', params)
c = cosmology.setCosmology('test')

z_arr = np.linspace(0, 10, 5)
try:
    t = c.lookbackTime(z_arr)
    print(f"Output type: {type(t)}")
    print(f"Output: {t}")
except Exception as e:
    print(f"Error: {e}")
