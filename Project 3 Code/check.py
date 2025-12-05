import sys
import numpy as np
import scipy
import matplotlib
import importlib

def try_import(name):
    try:
        module = importlib.import_module(name)
        return module.__version__
    except Exception as e:
        return f"Not installed or failed to import ({e})"

print("===== Python & Core Library Versions =====")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"matplotlib: {matplotlib.__version__}")

print("\n===== Machine Learning Library Versions =====")
print(f"TensorFlow: {try_import('tensorflow')}")
print(f"Keras: {try_import('keras')}")
print(f"PyTorch: {try_import('torch')}")
print(f"JAX: {try_import('jax')}")

print("\n===== Other Common Libraries =====")
print(f"h5py: {try_import('h5py')}")
print(f"pandas: {try_import('pandas')}")
print(f"xarray: {try_import('xarray')}")
print(f"tqdm: {try_import('tqdm')}")
