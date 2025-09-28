import importlib

_current_backend = None

def use_backend(backend: str):
    global _current_backend

    if backend == "numpy":
        _current_backend = importlib.import_module("numpy")
    elif backend == "cupy":
        _current_backend = importlib.import_module("cupy")
    else:
        raise ValueError(f"unsupported backend")

def active_backend():
    if _current_backend is None:
        try:
            use_backend("cupy")
        except ImportError:
            print("CuPy not available, using NumPy")
            use_backend("numpy")
            
    return _current_backend