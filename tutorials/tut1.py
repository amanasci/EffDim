import numpy as np

import effdim as ed

# Generating some random data
data = np.random.rand(1000, 700)

print("Data shape:", data.shape)

# Computing effective dimensionality
results = ed.compute_dim(data)

for key, value in results.items():
    print(f"{key}: {value}")
