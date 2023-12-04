from rns_io import *
from rns_model import StarDataFrame
import pandas as pd

StarData = StarDataFrame()
StarData = read_StarDataFrame_from_csv_file(StarData,"OneBitch")
StarData['eos_color'] = pd.Categorical(StarData['eos']).codes

max_values_multi = StarData.groupby('eos')[['rho_c', 'J']].max()
columns_to_normalize = ['rho_c', 'M', 'M_0', 'R', 'Omega', 'Omega_p', 'T/W', 'J', 'I', 'Phi_2',
                        'h_plus', 'h_minus', 'Z_p', 'Z_b', 'Z_f', 'omega_c/Omega', 'r_e', 'r_ratio', 'Omega_pa',
                        'Omega+', 'u_phi']

#for col in columns_to_normalize:
#    max_values = StarData.groupby('eos')[col].transform('max')
#    StarData[f'{col}_normalized'] = StarData[col] / max_values
#
#StarData['Angular_constant'] = (StarData['J'] / StarData['M']) / StarData['M']
#StarData['I_dimless'] = StarData['I'] / StarData['M']/StarData['M']/StarData['M']
#StarData['Phi_2_dimless'] = StarData['Phi_2'] / StarData['M']/StarData['M']/StarData['M']/StarData['J']/StarData['J']
#max_index = StarData['Phi_2_dimless'].idxmax()
#StarData.drop(max_index+1, axis=0)
#print(StarData.at[max_index, 'Phi_2_dimless'])
#print(StarData["Phi_2_dimless"])
#StarData['linear'] = np.linspace(0,1898,1899,axis=0)
#print(pd.Series(StarData['eos'].values.ravel()).nunique())
#StarData.plot.scatter(x='rho_c', y='M', c='eos_color', colormap='viridis', legend=False)
StarData.plot.scatter(x='rho_c', y='M')
plt.show()
