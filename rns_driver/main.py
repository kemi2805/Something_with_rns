import os
from rns_model import StarDataFrame
from solver import solve_star_sequence_max, solve_entire_series
from rns_io import *
from matplotlib import pyplot as plt

# A block that calculates and prints the maximum of an EOS
#directory = "/home/ken/Programme/rns/EOS/106/"
#StarData = StarDataFrame()
#counter = 0
#for filename in os.listdir(directory):
#    Filepath = directory + filename
#    print(Filepath)
#    clean_eos_data(Filepath)
#    StarData = solve_star_sequence_max(StarData, "../source/rns.v1.1d/rns", Filepath)
#    write_StarDataFrame_to_csv_file(StarData, "One_File")
#    counter = counter + 1 
#    if counter == 10:
#        break
#print(StarData)
#write_StarDataFrame_to_csv_file(StarData, "OneFiel")
#StarData.plot.scatter(x='rho_c', y='M')
#plt.show()

directory = "/home/ken/Programme/rns/EOS/106/"
StarData = StarDataFrame()
counter = 0
for filename in os.listdir(directory):
    Filepath = directory + filename
    print(Filepath)
    clean_eos_data(Filepath)
    StarData = solve_entire_series(StarData, "../source/rns.v1.1d/rns", Filepath)
    write_StarDataFrame_to_csv_file(StarData, "One_File")
    counter = counter + 1 
    if counter == 1:
        break
print(StarData)
write_StarDataFrame_to_csv_file(StarData, "OneBitch")
StarData.plot.scatter(x='rho_c', y='M')
plt.show()
