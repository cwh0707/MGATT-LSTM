import numpy as np
import pandas as pd

dist = np.random.rand(57, 57)  
np.save('city_distances.npy', dist)

np.random.seed(1000)
func = np.random.rand(57, 57)
np.save('city_functional.npy', func)

np.random.seed(4399)
neigh = np.random.randn(57, 57)
np.save('city_neighbor.npy', neigh)