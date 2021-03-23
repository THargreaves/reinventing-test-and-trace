from models import BaseModel


mod = BaseModel()
print('instantiation successful')

mod.simulate_data(N=10**4, P=6)
print('data simulation successful')
mod.run(iterations=100, warmup_iterations=500, chains=4)
print('running successful')

print(mod.runtime)
print(mod.mse)
print('absolute win')
