"""
Utilities for pyscf.dh

File structure
--------------

- 1st Layer: Code without involvement of PySCF in principle

  - ``general.py``: Infrastructure utilities. Should be definitions of classes used by
    most programs.
  - ``helper.py``: Simple helper functions.
  - ``interface.py``: Interfaces to PySCF or reflections. Should not call other 1st
    Layer functions or classes.

- 2nd Layer: Code with PySCF involvement but minimum involvement of 1st Layer

  - ``helper_pyscf.py``: Simple helper functions that requires PySCF.
  - ``dferi.py``: Helper additional functions for density fitting
"""

from .general import HybridDict, Params
from .helper import (
    calc_batch_size, gen_batch, gen_leggauss_0_1, gen_leggauss_0_inf, restricted_biorthogonalize,
    sanity_dimension, check_real, parse_incore_flag)
from .dferi import get_cderi_mo
from .interface import get_default_options

from .helper_pyscf import (
    parse_frozen_numbers, parse_frozen_list, parse_dh_xc_code_string)
