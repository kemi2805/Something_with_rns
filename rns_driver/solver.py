import subprocess
import numpy as np
import pandas as pd
from rns_model import StarDataFrame
from rns_io import *

#solves the star when you give it a central energy density and an angular moment
def solve_star_ang(sdf,rns_solver, Filepath, cen_energy, ang_moment, timeout=10):
    cmd = [rns_solver, 
           "-f", Filepath, 
           "-t", "jmoment", 
           "-e", str(cen_energy), 
           "-p", "2", 
           "-j", str(ang_moment)
           ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout = timeout)
    except subprocess.TimeoutExpired:
        print("Your C process took to long")
        return sdf
    lines = result.stdout.split()
    if ang_moment == 0:
        lines = [float(line) for line in lines[-20:-12]] + ["---"] + [0] + [float(line) for line in lines[-11:]]
        lines = [Filepath] + lines
    elif ang_moment != 0:
        lines = [float(line) for line in lines[-21:]]
        lines = [Filepath] + lines
    else:
        print("The ang_moment you gave into the function was very weird")
    star_value = StarDataFrame(lines)
    star_value = star_value.replace([np.inf, -np.inf], np.nan).dropna()
    star_value = StarDataFrame.concat([sdf, star_value], ignore_index=True)
    print(star_value)
    return star_value


def solve_star_gmass(sdf, rns_solver, Filepath, cen_energy, gmass, timeout = 10):
    cmd = [rns_solver,
           "-f", Filepath,
           "-t", "gmass",
           "-e", str(cen_energy),
           "-p", "2",
           "-m", str(gmass)
           ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("Your C process took to long")
        return sdf
    lines = result.stdout.split()
    if(np.allclose(cen_energy,float(lines[-20]))):
        lines = [float(line) for line in lines[-20:-12]] + ["---"] + [0] + [float(line) for line in lines[-11:]]
        lines = [Filepath] + lines
    elif(np.allclose(cen_energy, float(lines[-21]))):
        lines = [float(line) for line in lines[-21:]]
        lines = [Filepath] + lines
    else:
        print("Somehow the energy density doesn't line up, with what we gave it")
    print(lines)
    star_value = StarDataFrame(lines)
    star_value = star_value.replace([np.inf, -np.inf], np.nan).dropna()
    star_value = StarDataFrame.concat([sdf, star_value],  ignore_index=True)
    return star_value

 
# returning the maximal angular momentum as a float number, or if there is none 0
def calculate_maximum_star(rns_solver, Filepath, cen_energy):
    cmd = [rns_solver,
           "-f", Filepath,
           "-t", "kepler",
           "-e", str(cen_energy),
           "-p", "2"
           ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        print("C code took too long to run!")
        return 0
    lines = result.stdout.split()
    if(np.allclose(cen_energy,float(lines[-20]))):
        lines = [float(line) for line in lines[-20:-12]] + ["---"] + [0] + [float(line) for line in lines[-11:]]
        lines = [Filepath] + lines
    elif(np.allclose(cen_energy, float(lines[-21]))):
        lines = [float(line) for line in lines[-21:]]
        lines = [Filepath] + lines
    else:
        print("Somehow the energy density doesn't line up, with what we gave it")
    print(lines)
    star_value = StarDataFrame(lines)
    star_value = star_value.replace([np.inf, -np.inf], np.nan).dropna()
    return star_value

def calculate_maximum_angular_moment(rns_solver, Filepath, cen_energy):
    cmd = [rns_solver,
           "-f", Filepath,
           "-t", "kepler",
           "-e", str(cen_energy),
           "-p", "2"
           ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        print("C code took too long to run!")
        return 0
    lines = result.stdout.split()
    return float(lines[-14])

# Solve a star sequnce for various angular moments
# We define an intervall for the energy density
def solve_star_ang_sequence(sdf,rns_solver, Filepath, ang_moment, low_energy_density = 1e14, high_energy_density = 1e16, energy_density_delta = 0.1234e15, num_stars = 20):
    if energy_density_delta > (high_energy_density-low_energy_density)/(num_stars-1):
        energy_density_delta = (high_energy_density-low_energy_density)/(num_stars-1)

    for rho_c in np.arange(low_energy_density,high_energy_density, energy_density_delta):
        sdf = solve_star_ang(sdf,rns_solver, Filepath, rho_c, ang_moment)
        if sdf.shape[0] > 1:
            if not close_enough(sdf):
                row_number = sdf.index[-1]
                print("Floating Point Prescision go brrr")
                sdf = get_close_enough_ang(sdf, row_number-1, row_number)
    return sdf


#def solve_star_dimless_ang_sequence(sdf, J0sdf, rns_solver, Filepath, ang_moment, low_energy_density = 1e14, high_energy_density = 1e16, energy_density_delta = 0.1234e15, num_stars = 20):
#    Max_TOV_Star = StarDataFrame()
#    Max_TOV_Star = calculate_maximum_star(rns_solver, Filepath, TOV_energy_density)
#    Max_dimless_J = Max_TOV_Star.at[0,'J']/Max_TOV_Star.at[0,'M']/Max_TOV_Star.at[0,'M']
#    Max_TOV_Star = Max_TOV_Star.sort_values(by='M')
#
#    
#    if energy_density_delta > (high_energy_density-low_energy_density)/(num_stars-1):
#        energy_density_delta = (high_energy_density -
#                                low_energy_density)/(num_stars-1)
#
#    check = 0
#    for rho_c in np.arange(low_energy_density, high_energy_density, energy_density_delta):
#        sdf = solve_star_ang(sdf, rns_solver, Filepath, rho_c, ang_moment)
#        if sdf.shape[0] > 1:
#            if not close_enough(sdf):
#                row_number = sdf.index[-1]
#                print("Floating Point Prescision go brrr")
#                sdf = get_close_enough_ang(
#                    sdf, rns_solver, Filepath, 0, row_number-1, row_number)
#            row_number = sdf.shape[0]
#            if sdf.at[row_number-1, 'M'] < sdf.at[row_number-2, 'M']:
#                if check == 1:
#                    return sdf
#                check = 1
#    return sdf

def solve_star_ang_sequence_TOV(sdf, rns_solver, Filepath, ang_moment = 0, low_energy_density=1e14, high_energy_density=1e16, energy_density_delta=0.1234e15, num_stars=20):
    if energy_density_delta > (high_energy_density-low_energy_density)/(num_stars-1):
        energy_density_delta = (high_energy_density-low_energy_density)/(num_stars-1)

    check = 0
    for rho_c in np.arange(low_energy_density,high_energy_density, energy_density_delta):
        sdf = solve_star_ang(sdf,rns_solver, Filepath, rho_c, ang_moment)
        if sdf.shape[0] > 1:
            if not close_enough(sdf):
                row_number = sdf.index[-1]
                print("Floating Point Prescision go brrr")
                sdf = get_close_enough_ang(sdf,rns_solver, Filepath, 0, row_number-1, row_number)
            row_number = sdf.shape[0]
            if sdf.at[row_number-1, 'M'] < sdf.at[row_number-2, 'M']:
                if check == 1:
                    return sdf
                check = 1
    return sdf


def solve_star_sequence_max(sdf, rns_solver, Filepath, low_energy_density=1e14, high_energy_density=5e15, num_stars=20):
    temporary_star_data = StarDataFrame()
    temporary_star_data = solve_star_MTOV(sdf, rns_solver, Filepath, low_energy_density=0.5e15, high_energy_density=5e15,num_stars = 20)
    high_energy_density = temporary_star_data['rho_c'].iloc[-1]
    print("high energy density=",  high_energy_density)
    energy_density_delta = (high_energy_density - low_energy_density)/(num_stars-1)
    for i in range(num_stars):
        cen_density = (low_energy_density + i * energy_density_delta)
        ang_moment = calculate_maximum_angular_moment(rns_solver,Filepath,cen_density)
        print('ang moment =',ang_moment)
        if(ang_moment == 0 or ang_moment == np.nan):
            continue
        sdf = solve_star_ang(sdf, rns_solver, Filepath, cen_density, ang_moment)
    return sdf


def solve_entire_series(sdf, rns_solver, Filepath, low_energy_density=1e14, high_energy_density=5e16, num_stars=20, num_angular_mom = 10):
    temporary_star_data = StarDataFrame()
    Star_Data = StarDataFrame()

    temporary_star_data = solve_star_MTOV(temporary_star_data, rns_solver, Filepath, low_energy_density, high_energy_density,num_stars = 30)
    TOV_energy_density = temporary_star_data['rho_c'].iloc[-1]
    energy_density_delta = (TOV_energy_density - low_energy_density)/(num_stars-1)
    for rho_c in np.arange(low_energy_density, TOV_energy_density + energy_density_delta, energy_density_delta, dtype=float):
        Star_Data = solve_star_ang(Star_Data, rns_solver, Filepath, rho_c,0)
    sdf = StarDataFrame.concat([Star_Data, sdf], ignore_index=True)

    Max_TOV_Star = StarDataFrame()
    Max_TOV_Star = calculate_maximum_star(rns_solver, Filepath, TOV_energy_density)
    Max_dimless_J = Max_TOV_Star.at[0,'J']/Max_TOV_Star.at[0,'M']/Max_TOV_Star.at[0,'M']
    
    for j in range(1,num_angular_mom,1):
        temp_star_data = StarDataFrame()
        Dimless_J = j*Max_dimless_J/num_angular_mom
        for rho_c, M in zip(Star_Data['rho_c'], Star_Data['M']):
            J = Dimless_J*M*M
            temp_star_data = solve_star_ang(temp_star_data, rns_solver, Filepath, rho_c, J)
        sdf = StarDataFrame.concat([sdf, temp_star_data], ignore_index=True)
    return sdf


# Basically dividing by the square of the mass
# solve_star_ang needs the angular momentum with units
def make_angular_moment_dimensionless(StarData):
    StarData['J'] = (StarData['J']/StarData['M'])/StarData['M']



# This functions calculates the scenario with singular angular momentum and calculate the M_TOV
def solve_star_MTOV(sdf, rns_solver, Filepath, low_energy_density=1e14, high_energy_density=1e16,num_stars = 20):
    # Checks so that I do not have to calculate energy densities beyond the Tov Mass
    sdf = solve_star_ang_sequence_TOV(sdf, rns_solver, Filepath, 0, low_energy_density, high_energy_density, energy_density_delta=0.1234e15, num_stars=20)
    sdf = calculate_M_TOV(sdf, rns_solver, Filepath)
    return sdf


def calculate_M_TOV(sdf, rns_solver, Filepath):
    # Now we solve for the TOV-Star
    # Getting a fit for the upper region and then recieving a value for M_TOV
    rho_c = np.array(sdf['rho_c'].tail(4), dtype=float)
    gM = np.array(sdf['M'].tail(4), dtype=float)
    coefficients = np.polyfit(rho_c, gM, len(gM)-1)
    polynomial = np.poly1d(coefficients)

    # compute local minima
    # excluding range boundaries
    crit = polynomial.deriv().r
    print("crit =", crit)
    for value in crit:
        if value > rho_c[0] and value < rho_c[-1]:
            crit = value
            break
    r_crit = crit[crit.imag == 0].real
    test = polynomial.deriv(2)(r_crit)

    rho_TOV = r_crit[test < 0]
    M_TOV = polynomial(rho_TOV)
    try:
        print("rho_c =", rho_TOV[0])
        print("M_TOV =", M_TOV[0])
        sdf = solve_star_ang(sdf, rns_solver, Filepath,
                             rho_TOV[0], 0, timeout=30)
    except IndexError:
        print("Keine Ahnung wieso dieser IndexError kommt, wahrscheinlich zu wenige Datensätze für den Plot")
    return sdf

#returns 0 if not close enough and 1 when close enough
def close_enough(sdf, limit = 0.1234, x_norm = 1e15, y_norm=1):
    difference_squared = ((sdf['M'].iloc[-1]-sdf['M'].iloc[-2])/y_norm)**2 + ((sdf['rho_c'].iloc[-1]-sdf['rho_c'].iloc[-2])/x_norm)**2
    if difference_squared > 0.1234:
        return 0
    else:
        return 1
    

def get_close_enough_ang(sdf, rns_solver, Filepath, ang_moment, low_row, high_row, limit=0.1234, x_norm=1e15, y_norm=1):
    rho_c_check = 0
    M_check = 0
    for i in range(low_row,high_row,1):
        squared_difference = ((sdf.at[i+1, 'M'] - sdf.at[i, 'M'])/y_norm)**2+((sdf.at[i+1, 'rho_c'] - sdf.at[i, 'rho_c'])/x_norm)**2
        if(squared_difference < limit): continue

        new_rho = (sdf.at[i+1, 'rho_c'] - sdf.at[i, 'rho_c'])/2 + sdf.at[i, 'rho_c']
        sdf = solve_star_ang(sdf, rns_solver, Filepath, new_rho, ang_moment)
        sdf = sdf.sort_values(by='rho_c')
    if high_row != sdf.index[-1]:
        get_close_enough_ang(sdf, rns_solver, Filepath, ang_moment, low_row, sdf.index[-1])    
    return sdf




#StarData = StarDataFrame()
#StarData = solve_star_sequence_max(StarData, "../source/rns.v1.1d/rns", "/home/ken/Programme/rns/EOS/106/EOS86448.rns")
#write_StarDataFrame_to_csv_file(StarData, "One_File")
#print(StarData)
#StarData.plot.scatter(x='rho_c', y='M')
#plt.show()
