import colossus.halo.mass_so as mass_so
print(f"Dir(mass_so): {dir(mass_so)}")
if hasattr(mass_so, 'lagrangianRadius'):
    print("Found lagrangianRadius")
if hasattr(mass_so, 'M_to_R'):
    print("Found M_to_R")
