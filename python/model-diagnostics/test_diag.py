from models import BaseModel


mod = BaseModel()
print('instantiation successful')

mod.simulate_data(N=10**4, P=6)
print('data simulation successful')
mod.run(iterations=500, warmup_iterations=500, chains=1)
print('running successful')

print(mod.runtime)
print(mod.mse)
print('absolute win')
