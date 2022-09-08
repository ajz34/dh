"""
Utilities for pyscf.dh

File structure
--------------

- 1st Layer: Code without involvement of PySCF in principle

  - ``util_general.py``: Infrastructure utilities. Should be definitions of classes used by
    most programs.
  - ``util_helper.py``: Simple helper functions.
  - ``util_interface.py``: Interfaces to PySCF or reflections. Should not call other 1st
    Layer functions or classes.

- 2nd Layer: Code with PySCF involvement but minimum involvement of 1st Layer

  - ``util_dferi.py``: Helper additional functions for density fitting
"""

from .util_general import HybridDict, Params
from .util_helper import calc_batch_size, gen_leggauss_0_1, gen_leggauss_0_inf
from .util_dferi import get_cderi_mo
from .util_interface import default_options
