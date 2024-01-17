from model.NeutronStar import *
import sys
import os
from mpi4py import MPI
import math
from model.filter import *
from scipy import stats

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Define an eos
eos_folder = "/home/miler/codes/Something_with_rns/EOS/106"
#eos_folder = "/run/user/1001/gvfs/sftp:host=itp.uni-frankfurt.de,user=miler/home/miler/codes/Something_with_rns/test_eos_folder"
#eos_file_path = [f for f in os.listdir( eos_folder) if f.endswith('.rns')]
#eos_path = [os.path.join(eos_folder,f) for f in eos_file_path]

def main_parallel(eos_path):
    EosCatalog = NeutronStarEOSCatalog()
    print(rank)
    if rank == 0:  # Master process
            # Distribute EOS paths among worker processes
            print("TEST")
            for i, eos_path in enumerate(eos_path):
                print("i =",i)
                print("eos_path =", eos_path)
                if i % (size - 1) == 0:
                    continue  # Skip paths that belong to worker processes
                dest = 1 + (i % (size - 1))  # Distribute paths to worker processes
                print("preprocess")
                comm.Send(eos_path, dest=dest, tag=11)
                print("BAHAHHA")

            # Wait for worker processes to finish
            for worker_rank in range(1, size):
                comm.Recv(source=worker_rank, tag=22)
            print("Master process has finished processing EOS files.")
    else:  # Worker processes
        while True:
            print("TEST2")
            eos_path = comm.Recv(source=0, tag=11)
            print("Eyo")
            if eos_path is None:
                break  # No more work to do
            # Call _process_single_eos with the EOS path
            EosCatalog = EosCatalog._process_single_eos(eos_path)
            unique_ratio = df['r_ratio'].unique()
            rows_to_drop = []
            for ratio in unique_ratio:
                rows_to_drop = rows_to_drop + (filter_far_from_neighbors(df,ratio,1e15))
            print(rows_to_drop)
            print(df)
            df = df.drop(rows_to_drop)
            name = "hihihi" + str(rank) + ".csv"
            EosCatalog.df.to_parquet(name, index=False, mode='a')
            EosCatalog.df = None
            # Send a message back to the master indicating completion
            comm.Send(0, dest=0, tag=22)

def main_parallel_function(eos_folder):
    eos_path=None
    eos_path_buffer = None
    first_run = True
    if rank== 0:
        eos_file_path = [f for f in os.listdir(eos_folder) if f.endswith('.rns')]
        eos_path = [os.path.join(eos_folder,f) for f in eos_file_path]
        def split_list(lst, n):
            k, m = divmod(len(lst), n)
            return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
        eos_path = split_list(eos_path,size)
        eos_path = np.array(eos_path,dtype=object)
    eos_path = comm.scatter(sendobj=eos_path,root=0)
    if eos_path is None:
        return 0  # No more work to do
    if size == 1:
        for eos in eos_path:
            # Call _process_single_eos with the EOS path
            EosCatalog = NeutronStarEOSCatalog()
            EosCollection = EosCatalog._process_single_eos(eos)
            unique_ratio = EosCollection.df['r_ratio'].unique()
            rows_to_drop = []
            for ratio in unique_ratio:
                rows_to_drop = rows_to_drop + (filter_far_from_neighbors(df,ratio,1e15))
            df = df.drop(rows_to_drop)
            df.reset_index(drop=True, inplace=True)
            name = "hihihi_only_one.csv"
            EosCatalog.df.to_parquet(name, index=False, mode='a')
            EosCatalog.df = None
            EosCatalog = NeutronStarEOSCatalog()
            break
        return 0
    # else: not written out
    print("rank =", rank,"    EOSes =", eos_path)
    for eos in eos_path:
        print("--------------------------------------------------------")
        print("LOOOOOOOOK AAAATT MEEEEE")
        print(rank, eos)
        EosCatalog = NeutronStarEOSCatalog() # I had an error message, I think this will solve it
        EosCatalog = EosCatalog._process_single_eos(eos) # Generating all of the Stars and writing them onto a dataframe
        EosCatalog.df.reset_index(inplace=True) # Resetting potential index failures, because I want to be safe
        name = "/home/miler/codes/Something_with_rns/rns_driver/testEOS" + str(rank) + ".parquet"
        if first_run:
            EosCatalog.df.to_parquet(name, index=False, engine="fastparquet")
        else:
            EosCatalog.df.to_parquet(name, index=False, engine="fastparquet", append=True)
        unique_ratio = EosCatalog.df['r_ratio'].unique()
        rows_to_drop = []
        for ratio in unique_ratio:
            rows_to_drop = rows_to_drop + (filter_far_from_neighbors(EosCatalog.df,ratio,1e15)) # This finds the obviously wrong calculated stars
        EosCatalog.df = EosCatalog.df.drop(rows_to_drop) # This filters out the stars
        EosCatalog.df.reset_index(inplace=True) # We reset again, because the filtered data is somehow larger
        name = "/home/miler/codes/Something_with_rns/rns_driver/testEOSfiltered" + str(rank) + ".parquet"
        if first_run:
            EosCatalog.df.to_parquet(name, index=False, engine="fastparquet") # I had to write it like that, because the append method lates does not generate a new file
        else:
            EosCatalog.df.to_parquet(name, index=False, engine="fastparquet", append=True) # I am using fastparquet and not pyarrow, because fastparquet allows to append files
        EosCatalog.df = None # Python does not have a garbage disposal system (to my knowledge), so i do the next best thing. But I think it gets redundant with the next step
    return 0


main_parallel_function(eos_folder)
sys.exit()

#
#sys.exit()


#if __name__ == '__main__':
#    Test_dir = "/home/kenuni/Programs/rns/rns_driver2/Test"
#    EosCatalog = NeutronStarEOSCatalog()
#    EosCatalog.transmute_eos_file_to_200(Test_dir)
#    

eos = eos_path[0]
# Create a history list/ or rather simething else
history = NeutronStarEOSCollection(eos = eos)

# Define bounds for rho_c
a, b = 5e14, 8e15

def objective_function(rho_c, history, eos):
    ns = NeutronStar(eos = eos, rho_c = rho_c, J = 0)
    M = ns.M
    print("eos =", eos)
    print("rho_c",ns.rho_c)
    #print(rho_c,M)
    history.add_star(ns)
    return -M  # We're returning negative M because we want to maximize M

# Use brent to maximize M
result = minimize_scalar(objective_function, args=(history, eos), bracket = (a, b), method=custom_brent_polynomial, options={'maxiter': 15})
#history = [his for his in history if (his[0]>= a and his[0]<= b)]
optimal_rho_c = result.x
print(result)
print(f"Optimal rho_c for maximizing M is: {optimal_rho_c}")
print("And the maximized M is:", history.df.iloc[-1]["M"])

# Sort the DataFrame based on the 'rho_c' column
#sorted_history = history.df.sort_values(by='rho_c')

# Extract the 'rho_c' and 'M' values
#rho_c_values = sorted_history['rho_c'].tolist()
#M_values = sorted_history['M'].tolist()

Star_series = NeutronStarEOSCollection(eos = eos)
#Star_series.get_series(optimal_rho_c,1e13,{"J":0})
Star_series.traverse_r_ratio(optimal_rho_c, 0.05, initial_stepsize_rho_c = 1e12)

rho_c_values = Star_series.df['rho_c'].tolist()
M_values = Star_series.df['M'].tolist()
print(Star_series.df)
Star_series.df.to_csv("Hilbert.csv")

# Plotting
plt.scatter(rho_c_values, M_values, s=10)
plt.show()
