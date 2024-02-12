from model.NeutronStar import *
from model.filter import *
from _typeshed import Incomplete
from scipy import stats as stats

comm: Incomplete
rank: Incomplete
size: Incomplete
eos_folder: str

def main_parallel(eos_path) -> None: ...
def main_parallel_function(eos_folder): ...

eos: Incomplete
history: Incomplete
a: Incomplete
b: Incomplete

def objective_function(rho_c, history, eos): ...

result: Incomplete
optimal_rho_c: Incomplete
Star_series: Incomplete
rho_c_values: Incomplete
M_values: Incomplete
