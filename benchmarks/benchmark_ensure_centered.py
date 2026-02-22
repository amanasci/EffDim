import numpy as np
import time
from effdim.api import _ensure_centered

def benchmark():
    # Large data: 10000 samples, 1000 features
    n_samples = 10000
    n_features = 1000
    data = np.random.rand(n_samples, n_features)

    # Warmup
    _ensure_centered(data)

    start_time = time.time()
    for _ in range(100):
        _ensure_centered(data)
    end_time = time.time()

    print(f"Time taken for 100 iterations: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
