import os
import numpy as np
from scipy.io import loadmat

path = os.path.join("Data", "Input", "flapping_wing_aerodynamics.mat")
print("Loading:", path)

data = loadmat(path)
keys = [k for k in data.keys() if not k.startswith("__")]

print("\nTop-level variables in the .mat file:")
for k in keys:
    v = data[k]
    if hasattr(v, "shape"):
        print(f"  {k:35s} shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k:35s} type={type(v)}")

# Print a small sample from the first numeric variable
for k in keys:
    v = data[k]
    if hasattr(v, "shape") and np.size(v) > 0 and np.issubdtype(np.array(v).dtype, np.number):
        flat = np.array(v).reshape(-1)
        print(f"\nSample values from '{k}':")
        print(flat[:10])
        break