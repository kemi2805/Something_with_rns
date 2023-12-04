import pandas as pd


class StarDataFrame(pd.DataFrame):
    # define custom columns for each variable of the "rns" code
    _custom_columns = ['eos', 'rho_c', 'M', 'M_0', 'R', 'Omega', 'Omega_p', 'T/W', 'J', 'I', 'Phi_2',
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
        'J': "angular momentum in gravitational units cG/M^2",
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
        'Omega_pa': "There is no description yet",
        'Omega+': "There is no description yet",
        'u_phi': "There is no description yet"
    }

    def __init__(self, data=None, columns=_custom_columns, * args, **kwargs):
        if isinstance(data, list):
            data = [data]
        super().__init__(data, columns=columns, *args, **kwargs)
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
        return StarDataFrame(pd.concat(dfs, **kwargs))

    @staticmethod
    def read_csv(filename):
        # Custom function to read in csv
        return StarDataFrame(pd.read_csv(filename))

    @property
    def T(self):
        return StarDataFrame(self.transpose())
