# Make 'src' a package and re-export root-level config so that
# "from src import config" works when running from the project root.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
import config  # noqa: F401  – root-level config.py
