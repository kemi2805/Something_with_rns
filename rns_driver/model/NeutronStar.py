import sys
sys.path.append('/home/miler/codes/Something_with_rns/rns_driver')

import pandas as pd
from dataclasses import dataclass, asdict
from utils import rns_callback
from model.equations import custom_brent_polynomial
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
import numpy as np

import os
from concurrent.futures import ProcessPoolExecutor




custom_columns = ['eos', 'rho_c', 'M', 'M_0', 'R', 'Omega', 'Omega_p', 'T/W', 'J', 'I', 'Phi_2',
                       'h_plus', 'h_minus', 'Z_p', 'Z_b', 'Z_f', 'omega_c/Omega', 'r_e', 'r_ratio', 'Omega_pa',
                       'Omega+', 'u_phi']

rns_command = "/home/miler/codes/Something_with_rns/source/rns.v1.1d/rns"

@dataclass
class NeutronStar:
    eos: str = None
    rho_c: float = -1
    M: float = -1
    M_0: float = -1
    R: float = -1
    Omega: float = -1
    Omega_p: float = -1
    TW: float = -1
    J: float = -1
    I: float = -1
    Phi_2: float = -1
    h_plus: float = -1
    h_minus: float = -1
    Z_p: float = -1
    Z_b: float = -1
    Z_f: float = -1
    omega_c_over_Omega: float = -1
    r_e: float = -1
    r_ratio: float = -1
    Omega_pa: float = -1
    Omega_plus: float = -1
    u_phi: float = -1

    def __post_init__(self):
        # Call a function to compute/fetch the values based on rho_c and M
        if self.rho_c > 0:
            self.ComputeAttributes()
    
    def ComputeAttributes(self):
        #First we have to check what Values we we given
        if self.eos == None:
            print("You have to give me an eos")
            return None
        if self.rho_c < 0:
            print("You should always give me a central energy density")
            return None
        cmd = self.SettingCMD()
        StarValue = rns_callback.SolveNeutronStar(cmd)
        self.eos=self.eos.split("/")[-1]
        if StarValue:
            attributes = list(self.__annotations__.keys())
            for attr, value in zip(attributes, StarValue):
                setattr(self, attr, value)
            
    def is_valid(self):
        if self.R < 0 or self.M < 0 or self.M > 100:
            return False
        else:
            return True

    
    def SettingCMD(self):
        #Beginning of cmd implementation
        cmd = [rns_command,"-f",self.eos]
        if self.J > -0.1:
            cmd = cmd + ["-t", "jmoment", "-e", str(self.rho_c), "-j", str(self.J), "-p", str(2)]
        elif self.M > 0:
            cmd = cmd + ["-t", "gmass", "-e", str(self.rho_c), "-m", str(self.M), "-p", str(2)]
        elif self.M_0 > 0:
            cmd = cmd + ["-t", "rmass", "-e", str(self.rho_c), "-z", str(self.M_0), "-p", str(2)]
        elif self.Omega >= 0:
            cmd = cmd + ["-t", "gmass", "-e", str(self.rho_c), "-o", str(self.Omega), "-p", str(2)]
        elif self.r_ratio > 0:
            cmd = cmd + ["-t", "model", "-e", str(self.rho_c), "-r", str(self.r_ratio), "-p", str(2)]
        elif int(self.r_ratio) == 1:
            cmd = cmd + ["-t", "static", "-e", str(self.rho_c), "-p", str(2)]
        else:
            cmd = cmd + ["-t", "kepler", "-e", str(self.rho_c), "-b", str(1e-4), "-p", str(2)]
        return cmd
    
#This Class is used "collect" Stars from one ESO
class NeutronStarEOSCollection:
    def __init__(self, eos: str):
        self.eos = eos
        # Initialize an empty DataFrame with the columns we want.
        self.df = pd.DataFrame(columns=custom_columns)

    def add_star(self, star: NeutronStar):
        # Convert the NeutronStar dataclass to a dictionary 
        star_dict = asdict(star)

        # Create a new DataFrame from the dictionary
        star_df = pd.DataFrame([star_dict])

        # Concatenate the new DataFrame with the existing one
        self.df = pd.concat([self.df, star_df], ignore_index=True)

    def filter_by_mass(self, min_mass: float) -> pd.DataFrame:
        """Example method to filter stars by mass."""
        return self.df[self.df["M"] > min_mass]
    
    def filter_by_central_density(self, min_rho=-1.0, max_rho=-1.0, last_entry=False) -> pd.DataFrame:
        """Hold my beer... Let's filter these values!"""

        # If no boundaries are set, let's have a little fun!
        if min_rho < 0 and max_rho < 0:
            print("Why'd you call this if you're not filtering? Cheers, mate!")
            return self.df

        # Boundaries seem a bit off? No worries, I got you!
        if min_rho > max_rho:
            min_rho, max_rho = max_rho, min_rho

        # Craft the filter
        if min_rho >= 0 and max_rho >= 0:
            condition = self.df["rho_c"].between(min_rho, max_rho)
        elif min_rho >= 0:
            condition = self.df["rho_c"] >= min_rho
        else:
            condition = self.df["rho_c"] <= max_rho

        filtered_df = self.df[condition]

        # Just the last one? Say no more!
        if last_entry:
            return filtered_df.tail(1)
        return filtered_df

    def delete_outside_bounds(self, min_rho=-1.0, max_rho=-1.0, last_entry=False) -> None:
        """Method needed to get rid of bad values out of bounds."""
    
        # Case where no filtering is desired
        if min_rho < 0 and max_rho < 0:
            print("Why did you call this filter function if you do not filter? You deserved being filtered out of mankind!")
            return
    
        # Swap min_rho and max_rho if they are in the wrong order
        if min_rho > max_rho:
            min_rho, max_rho = max_rho, min_rho

        # Determine which rows to filter based on last_entry parameter
        rows_to_check = self.df.iloc[-1:] if last_entry else self.df
    
        # Apply filter based on conditions
        if min_rho >= 0 and max_rho >= 0:
            # ~ is inverting the series. Ture becomes False and False becomes True
            condition = ~rows_to_check["rho_c"].between(min_rho, max_rho)
        elif min_rho >= 0:
            condition = rows_to_check["rho_c"] < min_rho
        elif max_rho >= 0:
            condition = rows_to_check["rho_c"] > max_rho

        # Drop the rows based on the condition
        self.df.drop(rows_to_check[condition].index, inplace=True)

    def get_series(self, rho_TOV, initial_stepsize, options: dict):
        """Return a series of values based on the given options."""

        rho_c = rho_TOV
        previous_rho_c = None
        previous_M = None
                
        # Counter for consecutive invalid stars
        invalid_star_counter = 0

        # A bug could happen otherwise
        stepsize = initial_stepsize

        #predefine
        incline = 0 
        Star_count = 0
        

        while True:
            # Create a star with the current rho_c and options
            star = NeutronStar(eos=self.eos, rho_c=rho_c, **options)
            # If we have 3 consecutive invalid stars, break
            if invalid_star_counter == 3:
                break

            # Check if the formula returns a valid star
            if not star.is_valid():
                invalid_star_counter += 1
                rho_c -= stepsize
                continue
            else:
                invalid_star_counter = 0
            
            self.add_star(star)
            Star_count += 1
            
            # If this isn't our first iteration, adjust the step size based on incline
            if previous_rho_c is not None:
                if invalid_star_counter == 0:
                    incline = (star.M - previous_M) / ((rho_c - previous_rho_c))  # Change in M divided by change in rho_c
                    stepsize = initial_stepsize / (1 + abs(incline))
                elif invalid_star_counter > 0:
                    print("invalid_star_counter",invalid_star_counter)
                    stepsize = stepsize
                else:
                    stepsize = initial_stepsize
                
                ## Maybe makes sense as a filter
                #if(incline < 0):
                #    positive_incline_counter = 0
                #    index_to_drop = len(self.df) - 2
                #    self.df = self.df.drop(index_to_drop)
                #    Star_count -= 1

                # Some values somehow are weird. A bug in rns? This just gets rid of them.
                #if incline < 0:
                #    negative_incline_counter += 1
                #    if positive_incline_counter > 2:
                #        positive_incline_counter = 0
                #        index_to_drop = len(self.df) - 2
                #        self.df = self.df.drop(index_to_drop)
                #        print("I am sorry to inform you. rns bugged out")
                #elif incline > 0:
                #    positive_incline_counter += 1
                #    if positive_incline_counter > 10:
                #        negative_incline_counter = 0


            previous_rho_c = rho_c
            previous_M = star.M

            rho_c -= stepsize  # Decrementing rho_c to go towards zero

        self.df = self.df.reset_index(drop=True)
        if Star_count < 20:
            for i in range(Star_count):
                self.df = self.df.drop(len(self.df)-1)
        return self.df


    # Old idea:
    # I just use smaller r_ratio, until the program just stops. This gives me to many solutions, because my angular momentum 
    # becomes larger than j_kep and starts shedding mass
    # New idea:
    # I include a check, such that, while decreasing r_ratio, M has to grow. This has to be true for all eos.
    # Second idea:
    # I check what Omega_kep is with rns and just compare the values of them <---- What I did
    def traverse_r_ratio(self, rho_TOV, initial_stepsize_ratio, initial_stepsize_rho_c = 1e13):
        """This traverses every r_ratio and gets a series to each. This should be used"""
        ratio = 1 # R_ratio starts with 1 and then gets smaller
        previous_ratio = None
        rho_c = rho_TOV
        previous_rho_c = None

        #Need to have my break conditions
        invalid_collection_counter = 0

        # A bug could happen otherwise
        stepsize_ratio = initial_stepsize_ratio

        Stars = NeutronStarEOSCollection(self.eos)

        # Calculate entire series for no rotation first. Increasing the rotation incrementally
        
        while True: 
            lenght_of_dataframe_previously = len(self.df)
            if lenght_of_dataframe_previously != 0:
                star_check = NeutronStar(self.eos, rho_TOV, r_ratio=ratio)
                #star_mass = star_check.M
                #filtered_df = self.df[self.df['r_ratio'] == previous_ratio]
                #if star_mass <= filtered_df.loc[0,'M']:
                #    invalid_collection_counter += 1
                #    stepsize_ratio = stepsize_ratio/2.0
                #    ratio = previous_ratio + stepsize_ratio
                #    print("Let's see if this makes sense. I will have to check my angular momentum")
                omega_kepler = star_check.Omega
                filtered_df = self.df[self.df['r_ratio'] == previous_ratio]
                print(filtered_df)
                print("---------------------------------------------------")
                print("filtered_df.loc[lenght_of_dataframe_previously-1].at['Omega']",filtered_df.loc[lenght_of_dataframe_previously-1].at['Omega'])
                if filtered_df.loc[lenght_of_dataframe_previously-1].at['Omega'] > omega_kepler:
                    invalid_collection_counter += 1
                    stepsize_ratio = stepsize_ratio/2.0
                    ratio = previous_ratio + stepsize_ratio
                    print("Let's see if this makes sense. I will have to check my angular momentum")
            
            print("lenght_of_dataframe_previously", lenght_of_dataframe_previously)
            self.get_series(rho_TOV,initial_stepsize_rho_c,{"r_ratio": ratio})

            # If we have 6 consecutive invalid stars, break
            # it was previously 3 but I added a new statement about the Mass checking
            if invalid_collection_counter == 6:
                break

            # Check if the formula returns a valid star
            if lenght_of_dataframe_previously == len(self.df):
                invalid_collection_counter += 1
                stepsize_ratio = stepsize_ratio/2.0
                ratio = previous_ratio - stepsize_ratio
                continue
            else:
                invalid_star_counter = 0
            previous_ratio = ratio
            ratio -= stepsize_ratio       
        return self
    
            
            
class NeutronStarEOSCatalog:
    """This class should be useful for parallel processing. Splitting the eos to different processorss"""
    def __init__(self):
        self.eos_collections = {}  # Dictionary to hold NeutronStarEOSCollection objects

    def add_eos_collection(self, eos_name, eos_collection):
        """Add a NeutronStarEOSCollection for a specific EOS."""
        if eos_name in self.eos_collections:
            print(f"Warning: EOS named {eos_name} already exists. Overwriting.")
        self.eos_collections[eos_name] = eos_collection

    def get_eos_collection(self, eos_name):
        """Retrieve a NeutronStarEOSCollection by EOS name."""
        return self.eos_collections.get(eos_name, None)

    def list_all_eos(self):
        """List all available EOSs in the library. They are represented as path"""
        return list(self.eos_collections.keys())

    def _process_single_eos(self, eos_path):
        """Private method to process a single EOS file."""
        eos_collection = NeutronStarEOSCollection(eos_path)  # instantiate your EOSCollection here
        # Define bounds for rho_c
        a, b = 5e14, 8e15
        # Define a function, which maximum is it at M_TOV
        def objective_function(rho_c, history, eos):
            ns = NeutronStar(eos = eos, rho_c = rho_c, J = 0)
            M = ns.M
            print("eos =", eos)
            #print(rho_c,M)
            history.add_star(ns)
            return -M  # We're returning negative M because we want to maximize M

        # Use brent to maximize M
        result = minimize_scalar(objective_function, args=(eos_collection, eos_path), bracket = (a, b), method=custom_brent_polynomial, options={'maxiter': 30})
        optimal_rho_c = result.x
        print("The central energy density at TOV is", optimal_rho_c)
        eos_collection = NeutronStarEOSCollection(eos_path)
        eos_collection.traverse_r_ratio(optimal_rho_c, 0.5)
        rho_c_values = eos_collection.df['rho_c'].tolist()
        M_values = eos_collection.df['M'].tolist()
        print(eos_collection.df)
        return eos_collection 

    #def process_eos_directory(self, dir_path, max_workers=None):
        """Process all EOS files in a directory in parallel."""
        # Get a list of all EOS files in the directory
        eos_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.rns')]  
        print(eos_files)
        # Use ProcessPoolExecutor to distribute the EOS files across CPUs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self._process_single_eos, eos_files)


    def transmute_eos_file_to_200(self, dir_path):
        '''My rns code only takes in eos files with max 200 points. To achieve that, i delete the extra datapoints'''
        for filename in os.listdir(dir_path):
            # Step 1: Open file
            filename = dir_path + "/" + filename
            with open(filename, 'r') as data:
                rows = data.readlines()
                Size = int(rows[0])
                print(Size, filename)
                if Size <= 200:
                    continue
                ratio = np.ceil(Size/200)
            # Step 2: delete entries
            new_rows = []
            for i in np.arange(1, Size, ratio):
                new_rows.append(rows[int(i)])
            new_rows = [str(len(new_rows))+ "\n"] + new_rows
            #Step 3: Open file to write on
            with open(filename, 'w') as datei:
                datei.writelines(str(lines) for lines in new_rows)
        return 0






# Usage:
#
#star1 = NeutronStar(1.4, 10, 300)
#star2 = NeutronStar(2.1, 11, 280)
#
#collection = NeutronStarEOSCollection()
#collection.add_star(star1)
#collection.add_star(star2)
#
#print(collection.total_mass())  # 3.5
#print(collection.average_spin())  # 290.0
#print(collection.filter_by_mass(1.5))  # DataFrame with stars having mass > 1.5