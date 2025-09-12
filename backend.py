import importlib

_current_backend = None

def use_backend(backend: str):
    global _current_backend

    if backend == "numpy":
        _current_backend = importlib.import_module("numpy")
    elif backend == "cupy":
        _current_backend = importlib.import_module("cupy")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def active_backend():
    if _current_backend is None:
        use_backend("cupy")
        
    return _current_backend