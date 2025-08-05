from model.NeutronStar import *
import sys
import os
from mpi4py import MPI
from model.filter import *
from scipy import stats

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Define an eos
eos_folder = "/mnt/rafast/miler/codes/Something_with_rns/EOS/106"
# eos_folder = "/run/user/1001/gvfs/sftp:host=itp.uni-frankfurt.de,user
# =miler/home/miler/codes/Something_with_rns/test_eos_folder"
# eos_file_path = [f for f in os.listdir( eos_folder) if f.endswith('.rns')]
# eos_path = [os.path.join(eos_folder,f) for f in eos_file_path]


def main_parallel_function(eos_folder):
    eos_path = []
    first_run = True
    if rank == 0:
        eos_file_path = [f for f in os.listdir(eos_folder) if f.endswith(".rns")]
        eos_path = [os.path.join(eos_folder, f) for f in eos_file_path]

        def split_list(lst, n):
            k, m = divmod(len(lst), n)
            return [
                lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)
            ]

        eos_path = split_list(eos_path, size)
        #eos_path = np.array(eos_path, dtype=object)

    eos_path = comm.scatter(sendobj=eos_path, root=0)
    print("I am rank:", rank, "./ And i have the following eoses:",eos_path,"\n")
    if eos_path is None:
        return 0  # No more work to do
    if size == 1:
        for eos in eos_path:
            # Call _process_single_eos with the EOS path
            EosCatalog = NeutronStarEOSCatalog()
            EosCollection = EosCatalog._process_single_eos(eos)
            unique_ratio = EosCollection.df["r_ratio"].unique() # Hier ist alles durcheinander. Was hab ich mir gedacht?
            rows_to_drop = []
            for ratio in unique_ratio:
                rows_to_drop = rows_to_drop + (
                    filter_far_from_neighbors(EosCollection.df, ratio, 1e15)
                )
            df = EosCollection.df.drop(rows_to_drop)
            df.reset_index(drop=True, inplace=True)
            name = "hihihi_only_one.csv"
            EosCollection.df.to_parquet(name, index=False, mode="a")
            EosCollection.df = pd.DataFrame()
            EosCatalog = NeutronStarEOSCatalog()
        return 0
    # else: not written out
    print("rank =", rank, "    EOSes =", eos_path)
    for eos in eos_path:
        print("--------------------------------------------------------")
        print("LOOOOOOOOK AAAATT MEEEEE")
        print(rank, eos)
        EosCatalog = NeutronStarEOSCatalog()
        # Generating all of the Stars and writing them onto a dataframe
        EosCollection = EosCatalog._process_single_eos(eos)
        # Resetting potential index failures, because I want to be safe
        EosCollection.df.reset_index(inplace=True)
        name = (
            "/mnt/rafast/miler/codes/Something_with_rns/rns_driver/testEOS"
            + str(rank)
            + ".parquet"
        )
        if first_run:
            EosCollection.df.to_parquet(name, index=False, engine="fastparquet")
            # Auskommentieren wenn die Zeit kommt
            #first_run = False
        else:
            EosCollection.df.to_parquet(
                name, index=False, engine="fastparquet", append=True
            )
        unique_ratio = EosCollection.df["r_ratio"].unique()
        rows_to_drop = []
        for ratio in unique_ratio:
            # This finds the obviously wrong calculated stars
            rows_to_drop = rows_to_drop + (
                filter_far_from_neighbors(EosCollection.df, ratio, 1e15)
            )
        EosCollection.df = EosCollection.df.drop(
            rows_to_drop)  # This filters out the stars
        # We reset again, because the filtered data is somehow larger
        EosCollection.df.reset_index(inplace=True)
        name = (
            "/mnt/rafast/codes/Something_with_rns/rns_driver/testEOSfiltered"
            + str(rank)
            + ".parquet"
        )
        if first_run:
            # I had to write it like that, because the append method lates does not generate a new file
            EosCollection.df.to_parquet(name, index=False, engine="fastparquet")
            first_run = False
        else:
            # I am using fastparquet and not pyarrow, because fastparquet allows to append files
            EosCollection.df.to_parquet(
                name, index=False, engine="fastparquet", append=True
            )
        # Python does not have a garbage disposal system (to my knowledge), so i do the next best thing. But I think it gets redundant with the next step
        EosCollection.df = pd.DataFrame()
    return 0


main_parallel_function(eos_folder)
sys.exit()
