import matplotlib.pyplot as plt
import sys
import subprocess
import numpy as np
import pandas as pd

#define panda array, which captures the entire rns code.
# 'eos': The current eos being used
# 'rho_c' central energy density
# 'M' gravitational mass
# 'M_0' rest mass
# 'R' radius at the equator
# 'Omega' angular velocity
# 'Omega_p' angular velocity of a particle in circular orbit at the equator
# 'T/W' rotational/gravitational energy
# 'C*J/GM^2' angular momentum
# 'I' moment of inertia (except for nonrotating model)
# 'h_plus' height from surface of last stable co-rotating circular orbit in equatorial plane (circumferencial) - if none, then all such orbits are stable
# 'h_minus' height from surface of last stable counter-rotating circular orbit in equatorial plane (circumferential) - if none, then all such orbits are stable
# 'Z_p' polar redshift
# 'Z_b' backward equatorial redshift
# 'Z_f' forward equatorial redshift
# 'omega_c/Omega' ratio of central value of potential ω to Ω
# 'r_e' coordinate equatorial radius
# 'r_ratio' axes ratio
# 'Omega_pa'
# 'Omega+'
# 'u_phi'


class star_dataframe(pd.DataFrame): 
    #define custom columns for each variable of the "rns" code
    _custom_columns = ['eos', 'rho_c', 'M', 'M_0', 'R', 'Omega', 'Omega_p', 'T/W', 'cJ/GM^2', 'I', 'Phi_2',
                       'h_plus', 'h_minus', 'Z_p', 'Z_b', 'Z_f', 'omega_c/Omega', 'r_e', 'r_ratio', 'Omega_pa',
                       'Omega+', 'u_phi']
    _column_descriptions = {
        'eos': "The current eos being used",
        'rho_c': "central energy density",
        'M': "gravitational mass",
        'M_0': "rest mass",
        'R': "radius at the equator",
        'Omega': "angular velocity",
        'Omega_p': "angular velocity of a particle in circular orbit at the equator",
        'T/W': "rotational/gravitational energy",
        'C*J/GM^2': "angular momentum",
        'I': "moment of inertia (except for nonrotating model)",
        'Phi_2': "quadrupole moment(program needs to be compiled on HIGH resolution for this to beaccurate)",
        'h_plus': "height from surface of last stable co-rotating circular orbit in equatorial plane (circumferencial) - if none, then all such orbits are stable",
        'h_minus': "height from surface of last stable counter-rotating circular orbit in equatorial plane (circumferential) - if none, then all such orbits are stable",
        'Z_p': "polar redshift",
        'Z_b': "backward equatorial redshift",
        'Z_f': "forward equatorial redshift",
        'omega_c/Omega': "ratio of central value of potential ω to Ω",
        'r_e': "coordinate equatorial radius",
        'r_ratio': "axes ratio",
        'Omega_pa':"There is no description yet",
        'Omega+': "There is no description yet",
        'u_phi':"There is no description yet"   
    }

    def __init__(self, data=None, columns=_custom_columns, * args, **kwargs):
        if isinstance(data,list):
            data = [data]
        super().__init__(data, columns = columns, *args, **kwargs)
        # Transpose it, because it is a lot easier working like this
        # Apply the custom column names
        if len(self.columns) == len(self._custom_columns):
            self.columns = self._custom_columns

    def describe_column(self, column_name):
        """Get the description for a given column."""
        return self._column_descriptions.get(column_name, "Description not available.")

    @staticmethod
    def concat(dfs, **kwargs):
        # Your custom concat behavior
        # For simplicity, we'll just call pandas' concat function here,
        # but you can introduce custom behavior if desired.
        return star_dataframe(pd.concat(dfs, **kwargs))
    
    @staticmethod
    def read_csv(filename):
        # Custom function to read in csv
        return star_dataframe(pd.read_csv(filename))
    
    @property
    def T(self):
        return star_dataframe(self.transpose())

#Some global variables, which I have defined as
#######################################
Path_to_rns_code = "./rns"
saturation_density = 2e14
maximal_energy_density = 8e15 # I do not have good values for these
star_data = star_dataframe() #Type ist star_dataframe
#######################################


def retrieve_eos_files(filepath, eos_paths = []):
    if eos_paths: 
        print("Your list is not empty")
        print("Will abort")
        return 1
    with open(filepath, 'r') as eosfile:
        for line in eosfile:
            eos_paths.append(line.replace("\n",""))
    return 0

def calculate_eos_sequence(filepath):
    return 0

def calculate_star_sequence_by_cen_energy(filepath, low_cen_energy = saturation_density, high_cen_energy = maximal_energy_density, ang_moment = 0, star_num = 10):
    #We start to calculate the star without angular moment. 
    cen_energy = 1e15
    cen_energy_delta = (high_cen_energy - low_cen_energy)/(star_num-1)
    for i in range(star_num):
        cen_energy = (low_cen_energy + i * cen_energy_delta)
        result = calculate_star(filepath, cen_energy, ang_moment)
        if result == 1:
            continue
        lines = result.stdout.split()
        if(ang_moment==0):
            lines = [(line) for line in lines[-20:-11]] + [0] + [(line) for line in lines[-11:]] 
        else:
            lines = [(line) for line in lines[-21:]]
        lines = [filepath] + lines
        star_value = star_dataframe(lines)

        global star_data 
        star_data = star_dataframe.concat([star_data,star_value])
    return 0


def calculate_star_sequence_by_ang_moment(filepath, cen_energy = 1e15, num_of_seq=20):
    ang_max = maximum_angular_momentum(filepath,cen_energy)
    if(ang_max==None):
        return 0
    global star_data
    ang_delta = ang_max/(num_of_seq-1)
    result = calculate_star(filepath, cen_energy, 0)
    if result == 1: 1
    else:
        lines = result.stdout.split()
        lines = [(line) for line in lines[-20:-11]] + [0] + [(line) for line in lines[-11:]]
        lines = [filepath] + lines
        star_value = star_dataframe(lines)
        star_data = star_dataframe.concat([star_data, star_value])
    for i in range(1,num_of_seq):
        print(i)
        result = calculate_star(filepath, cen_energy, i*ang_delta)
        if result == 1:
            continue
        lines = result.stdout.split()
        lines = [(line) for line in lines[-21:]]
        lines = [filepath] + lines
        star_value = star_dataframe(lines)
        star_data = star_dataframe.concat([star_data, star_value])
    return 0

def calculate_star(filepath, cen_energy, ang_moment):
    try:
        result = subprocess.run([Path_to_rns_code, "-f", filepath, "-t", "jmoment", "-e", str(
            cen_energy), "-p", "2", "-j", str(ang_moment)], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        print("Your C process took to long")
        return 1
    return result


#def mean_angular_momentum(func):
#    ang_moment_list = []
#    def mean_calc(filepath, energy_densities):
#        for energy_density in energy_densities:
#            temp = func(filepath, energy_density)
#            if temp != 0:
#                ang_moment_list.append(temp)
#        mean = sum(ang_moment_list)/len(ang_moment_list)
#        print(ang_moment_list)
#        return mean
#    return mean_calc

#@mean_angular_momentum
def maximum_angular_momentum(filepath, energy_density):
    try:
        result = subprocess.run([Path_to_rns_code, "-f", filepath, "-t", "kepler", "-e", str(
            energy_density), "-p", "2"], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        print("C code took too long to run!")
        return None
    lines = result.stdout.split()
    return float(lines[-14]) # lines[-17] should be the angular momentum in the rns framework

def central_difference_derivative(point_a, point_b, h):
    return (point_b-point_a)/(2*h)
def central_difference_derivative(list_y = [], list_x = [], point = 1):
    if point == 0:
        return (list_y[1]-list_y[0])/(list_x[1]-list_x[0])
    elif point == (len(list_y)-1):
        return (list_y[point]-list_y[point-1])/(list_x[point]-list_x[point-1])
    else:
        return (list_y[point+1]-list_y[point-1])/(list_x[point+1]-list_x[point-1])
    
def write_panda_to_csv_file(dataframe):
    try:
        dataframe.to_csv('stars.csv', index=False)
    except FileNotFoundError:
        print("We could not find the right dirctory")
        return 1
    except PermissionError:
        print("We did not have the permission to write into the file")
        return 1
    return 0


#for something in np.arange(1e15,1e16,(1e16-1e15)/19):
#    calculate_star_sequence_by_ang_moment("eos/eosA",cen_energy=something)
#star_data.to_csv("Some_eosA")
#calculate_star_sequence_by_ang_moment("eos/eosA")
#calculate_star_sequence_by_ang_moment("eos/eosA",5e15)
star_data = star_dataframe.read_csv("Some_eosA")
print("test test test")
star_data = star_data.replace([np.inf,-np.inf],np.nan).dropna()
print(star_data)
#mass_data = star_dataframe(star_data[(star_data['cJ/GM^2'] == 0e15)])
#print(star_data[['M']])
#print(star_data[['cJ/GM^2']].max())
mass_data = star_data["M"]
ang_data = star_data["cJ/GM^2"]/mass_data
ang_data = ang_data/mass_data
rho_data = star_data["rho_c"]

plt.scatter(rho_data,ang_data)
#star_data.plot.scatter(x="rho_c", y="")
plt.show()

#print(star_data[['rho_c', 'C*J/GM^2', 'Phi_2']])

#cen_energy = 6e15
#min_cen_energy = 5.7e15
#max_cen_energy = 5.85e15
#energy_densities = []
#angular_moments = []
#for i in range(10):
#    energy_densities.append(min_cen_energy + i*(max_cen_energy-min_cen_energy)/9)
#
#for energy in energy_densities:
#    angular_moments.append(maximum_angular_momentum("eos/eosC", energy))
#
#plt.scatter(energy_densities,angular_moments)
#plt.show()
#
#print(maximum_angular_momentum("eos/eosC", energy_densities))

