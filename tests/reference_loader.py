"""Load the frozen pre-refactor modules without contaminating production imports."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def load_reference():
    directory = Path(__file__).parent / "reference"
    names = ("data", "neighbors", "scope", "models", "diffusion", "train")
    previous = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            alias = "_grow_reference_" + name
            spec = importlib.util.spec_from_file_location(alias, directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return SimpleNamespace(**loaded)
