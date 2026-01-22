import numpy as np

import effdim as ed

# Generating some random data
data = np.random.rand(1000, 700)


# Generating data such that some dimensions are more important than others
new_data = np.random.rand(1000, 10)  # 10 important dimensions
data[:, :10] += new_data * 5  # Amplify the important dimensions

print("Data shape:", data.shape)

# Computing effective dimensionality
results = ed.compute_dim(data)

for key, value in results.items():
    print(f"{key}: {value}")
