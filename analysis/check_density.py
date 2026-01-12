from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.96}
cosmology.addCosmology('test', params)
c = cosmology.setCosmology('test')
print(f"rho_m(0): {c.rho_m(0.0)}")
print(f"rho_c(0): {c.rho_c(0.0)}")
