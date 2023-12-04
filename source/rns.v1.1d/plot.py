import matplotlib.pyplot as plt
import sys
import subprocess
import numpy as np

path_to_eos = "eos/eosA"

cen_energy = 1e15
min_cen_energy = 1.2e13
max_cen_energy = 7e15

number_of_stars = 30
star_values = []
result =  subprocess

log_energy_discrete = (np.log(max_cen_energy) - np.log(min_cen_energy))/number_of_stars
energy_discrete = (max_cen_energy - min_cen_energy)/number_of_stars

for i in range(number_of_stars+1):
    cen_energy = min_cen_energy + i * energy_discrete
    print("We are on iteration", i)
    print("We have a value for our energy density of", cen_energy)
    try:
        result = subprocess.run(["./rns", "-f", path_to_eos, "-t", "jmoment", "-j", "0", "-e", str(cen_energy), "-p", "2"], capture_output=True, text=True, timeout = 10)
    except subprocess.TimeoutExpired:
        print("C code took too long to run!")
    else:
        lines = result.stdout.split()
        star_values.append([(line) for line in lines[-20:]])
    print("")


energy_densities = [float(sublist[0]) for sublist in star_values if (np.isfinite(float(sublist[0])) and float(sublist[0]) < max_cen_energy)]
star_masses = [float(sublist[1]) for sublist in star_values if (np.isfinite(float(sublist[1])) and float(sublist[0]) < max_cen_energy)]
star_radius = [float(sublist[3]) for sublist in star_values if (np.isfinite(float(sublist[3])) and float(sublist[0]) < max_cen_energy)]
axes_ratio = [float(sublist[16]) for sublist in star_values if (np.isfinite(float(sublist[16])) and float(sublist[0]) < max_cen_energy)]
#print(energy_densities)
#print(star_masses)

plt.scatter(energy_densities,star_masses)
plt.show()
print(star_masses)
print(axes_ratio)
print(star_radius)
plt.scatter(star_masses, axes_ratio)
plt.show()

plt.scatter(star_radius, star_masses)
plt.show()



