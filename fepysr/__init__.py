"""
FePySR: A Two-Stage Symbolic Regression Framework via Feature Engineering
"""


from .fepysr import FePySR
from .data_analyzer import DataAnalyzer

__version__ = "0.1.0"

__all__ = [
    "FePySR",
    "DataAnalyzer",
    "__version__"
]