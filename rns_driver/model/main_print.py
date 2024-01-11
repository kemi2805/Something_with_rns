from NeutronStar import *
from mpi4py import MPI
from scipy import stats
from filter import *

#Plotting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_parquet('/home/miler/codes/Something_with_rns/rns_driver/hihihi0.parquet')
#df.to_parquet('/home/miler/codes/Something_with_rns/rns_driver/Hihi(copy).parquet', engine="pyarrow")

unique_z = df['r_ratio'].unique()
unique_ratio = df['r_ratio'].unique()
rows_to_drop = []
for ratio in unique_ratio:
    rows_to_drop = rows_to_drop + (filter_far_from_right_neighbors(df,ratio,5e-3))
    ratio_df = df[df['r_ratio'] == int(ratio)]
df = df.drop(rows_to_drop)
df.reset_index(drop=True, inplace=True)

colormap = []
for i in range(len(unique_z)):
    colormap.append((i)/len(unique_z))
colors = plt.cm.Blues(colormap)

# Create a scatter plot for each distinct curve based on 'z' value
for z_value, color in zip(unique_z, colors):
    curve = df[df['r_ratio'] == z_value]
    rho_values = curve['rho_c'].tolist()
    M_values = curve['M'].tolist()
    plt.scatter(rho_values, M_values, color=[color], label=f'Curve {z_value}')
plt.show()