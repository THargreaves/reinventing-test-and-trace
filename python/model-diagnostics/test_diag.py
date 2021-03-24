import diagnostics
from models import BaseModel


mod = BaseModel()
print('instantiation successful')
a = 10**4
Ns = [5*(10**5), a, 5*a, 10*a,50*a ,100*a]
diagnostics.runtime_lineplot_N(mod, N_space = Ns, P = 4)
print('ahuevooooooooo')
