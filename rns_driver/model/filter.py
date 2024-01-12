import math
import numpy as np

def eos_filter(df, fix_value_dict):
    '''Some points do not fit on the curve. I will disregard them'''
    def filter_fix_values(df, fix_values):
        for key, value in fix_value_dict:
            if(math.isclose(df[key], value)):
                return True
            else: 
                return False
            
    rows_to_remove = []
    for index, row in df.iterrows():
        if index > 0 and (math.isclose(row['r_ratio'],df.at[index+1])):
            if (row['rho_c'] > df.at[index -1, 'rho_c'] and row['rho_c'] > df.at[index +1, 'rho_c']) or ():
                return True

def filter_rho_c_far_from_neighbors(df, constant_r_ratio, tolerance):
    # Filter the DataFrame for the constant 'r_ratio'
    filtered_df = df[df['r_ratio'] == constant_r_ratio]
    
    if filtered_df.empty:
        # No matching 'r_ratio' values found
        return []
    far_indices = []
    
    for index in filtered_df.index:
        rho_c_value = filtered_df.loc[index, 'rho_c']

        if index > filtered_df.index[0] and index == filtered_df.index[0]:
            right_neighbor = filtered_df.loc[index - 1, 'rho_c']
            left_neighbor = filtered_df.loc[index + 1, 'rho_c']
            if abs(rho_c_value - right_neighbor) > tolerance and abs(rho_c_value - left_neighbor) > tolerance:
                far_indices.append(index)
               
        elif index == filtered_df.index[0]:
            left_neighbor = filtered_df.loc[index + 1, 'rho_c']
            if abs(rho_c_value - left_neighbor) > tolerance:
                far_indices.append(index)
        elif index == filtered_df.index[-1]:
            right_neighbor = filtered_df.loc[index - 1, 'rho_c']
            if abs(rho_c_value - right_neighbor) > tolerance:
                far_indices.append(index)
    return far_indices

def filter_far_from_neighbors(df, constant_r_ratio, tolerance):
    # Filter the DataFrame for the constant 'r_ratio'
    filtered_df = df[df['r_ratio'] == constant_r_ratio]
    
    if filtered_df.empty:
        # No matching 'r_ratio' values found
        return []
    filtered_df = filtered_df.reset_index(drop=True)

    far_indices = []
    rho_max = df['rho_c'][0]
    M_max = df['M'].max()
    for index in filtered_df.index:
        rho_c_value = filtered_df.loc[index, 'rho_c']/rho_max
        M_value = filtered_df.loc[index, 'M']/M_max

        if index > filtered_df.index[0] and index < filtered_df.index[-1]:
            right_neighbor_rho = filtered_df.loc[index - 1, 'rho_c']/rho_max
            left_neighbor_rho = filtered_df.loc[index + 1, 'rho_c']/rho_max
            right_neighbor_M = filtered_df.loc[index - 1, 'M']/M_max
            left_neighbor_M = filtered_df.loc[index + 1, 'M']/M_max
            if np.sqrt(abs(rho_c_value - right_neighbor_rho)**2 + abs(M_value - right_neighbor_M)**2)  > tolerance or np.sqrt(abs(rho_c_value - left_neighbor_rho)**2 + abs(M_value - left_neighbor_M)**2) > tolerance:
                far_indices.append(index)
               
        elif index == filtered_df.index[0]:
            left_neighbor_rho = filtered_df.loc[index + 1, 'rho_c']/rho_max
            left_neighbor_M = filtered_df.loc[index + 1, 'M']/M_max
            if np.sqrt(abs(rho_c_value - left_neighbor_rho)**2 + abs(M_value - left_neighbor_M)**2)  > tolerance:
                far_indices.append(index)
        elif index == filtered_df.index[-1]:
            right_neighbor_rho = filtered_df.loc[index - 1, 'rho_c']/rho_max
            right_neighbor_M = filtered_df.loc[index - 1, 'M']/M_max
            if np.sqrt(abs(rho_c_value - right_neighbor_rho)**2 + abs(M_value - right_neighbor_M)**2) > tolerance:
                far_indices.append(index)
    return far_indices

def filter_M_far_from_neighbors(df, constant_r_ratio, tolerance):
    # Filter the DataFrame for the constant 'r_ratio'
    filtered_df = df[df['r_ratio'] == constant_r_ratio]
    
    if filtered_df.empty:
        # No matching 'r_ratio' values found
        return []
    far_indices = []
    
    for index in filtered_df.index:
        rho_c_value = filtered_df.loc[index, 'rho_c']
        M_value = filtered_df.loc[index, 'M']

        if index > filtered_df.index[0] and index < filtered_df.index[-1]:
            right_neighbor_rho = filtered_df.loc[index - 1, 'rho_c']
            left_neighbor_rho = filtered_df.loc[index + 1, 'rho_c']
            right_neighbor_M = filtered_df.loc[index - 1, 'M']
            left_neighbor_M = filtered_df.loc[index + 1, 'M']
            test1 = (right_neighbor_M-left_neighbor_M)/(right_neighbor_rho-left_neighbor_rho)
            test2 = (M_value-left_neighbor_M)/(rho_c_value-left_neighbor_rho)
            test3 = ((right_neighbor_M-left_neighbor_M)/(right_neighbor_rho-left_neighbor_rho)-(M_value-left_neighbor_M)/(rho_c_value-left_neighbor_rho))
            test4 = ((right_neighbor_M-left_neighbor_M)/(right_neighbor_rho-left_neighbor_rho)-(M_value-left_neighbor_M)/(rho_c_value-left_neighbor_rho))/(M_value-left_neighbor_M)/(rho_c_value-left_neighbor_rho)
            if ((right_neighbor_M-left_neighbor_M)/(right_neighbor_rho-left_neighbor_rho)-(M_value-left_neighbor_M)/(rho_c_value-left_neighbor_rho))/(M_value-left_neighbor_M)/(rho_c_value-left_neighbor_rho) > tolerance:
                far_indices.append(index)
               
        elif index == filtered_df.index[0]:
            left_neighbor_rho = filtered_df.loc[index + 1, 'rho_c']
            left_neighbor_M = filtered_df.loc[index + 1, 'M']*5e14
            if np.sqrt(abs(rho_c_value - left_neighbor_rho)**2 + abs(M_value - left_neighbor_M)**2)  > tolerance:
                far_indices.append(index)
        elif index == filtered_df.index[-1]:
            right_neighbor_rho = filtered_df.loc[index - 1, 'rho_c']
            right_neighbor_M = filtered_df.loc[index - 1, 'M']*5e14
            if np.sqrt(abs(rho_c_value - right_neighbor_rho)**2 + abs(M_value - right_neighbor_M)**2) > tolerance:
                far_indices.append(index)
    return far_indices


# No star can exist, below the curve of non rotating stars
def filter_below_non_rotating(df, tolerance, constant_r_ratio = 0):    
    stationary_df = df[df['r_ratio'] == 0.0]
    if df.empty:
        # No matching 'r_ratio' values found
        return []
    far_indices = []
    
    for index in df.index:

        rho_c_value = df.loc[index, 'rho_c']
        M_value = df.loc[index, 'M']


def filter_far_from_right_neighbors(df, constant_r_ratio, tolerance):
    # Filter the DataFrame for the constant 'r_ratio'
    filtered_df = df[df['r_ratio'] == constant_r_ratio]
    
    if filtered_df.empty:
        # No matching 'r_ratio' values found
        return []
    far_indices = []
    rho_max = df['rho_c'][0]
    M_max = df['M'].max()
    good_index = filtered_df.index[0] # We need, so we have useful values to compare to
    preferred_tolerance = tolerance # If a point gets skipped the tolerance has to keep up and be bigger
    for index in filtered_df.index:
        rho_c_value = filtered_df.loc[index, 'rho_c']/rho_max
        M_value = filtered_df.loc[index, 'M']/M_max

        if index > filtered_df.index[0] and index < filtered_df.index[-1]:
            if index - good_index > 1:
                right_neighbor_rho = filtered_df.loc[good_index, 'rho_c']/rho_max
                right_neighbor_M = filtered_df.loc[good_index, 'M']/M_max
                preferred_tolerance = tolerance*(index - good_index)
            else:
                right_neighbor_rho = filtered_df.loc[index - 1, 'rho_c']/rho_max
                right_neighbor_M = filtered_df.loc[index - 1, 'M']/M_max
            if np.sqrt(abs(rho_c_value - right_neighbor_rho)**2 + abs(M_value - right_neighbor_M)**2)  > preferred_tolerance:
                far_indices.append(index)
                continue
               
        elif index == filtered_df.index[0]:
            left_neighbor_rho = filtered_df.loc[index + 1, 'rho_c']/rho_max
            left_neighbor_M = filtered_df.loc[index + 1, 'M']/M_max
            if np.sqrt(abs(rho_c_value - left_neighbor_rho)**2 + abs(M_value - left_neighbor_M)**2)  > tolerance:
                far_indices.append(index)
                continue

        elif index == filtered_df.index[-1]:
            right_neighbor_rho = filtered_df.loc[index - 1, 'rho_c']/rho_max
            right_neighbor_M = filtered_df.loc[index - 1, 'M']/M_max
            if np.sqrt(abs(rho_c_value - right_neighbor_rho)**2 + abs(M_value - right_neighbor_M)**2) > tolerance:
                far_indices.append(index)
                continue

        preferred_tolerance = tolerance
        good_index = index
    return far_indices
