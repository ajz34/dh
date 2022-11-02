# region General Options

print_level_energy = 0
""" Print level for pyscf.dh.energy.

Value of this option is related to PySCF's print level.
"""

# endregion

# region Molecular specific

frozen_rule = None
""" Rule for frozen orbital numbers.

This option will be generate and be overrided by option ``frozen_list``.

Warnings
--------
TODO: Function to parse frozen rule.
"""

frozen_list = None
""" Index list of frozen orbitals.

For example, if set to ``[0, 2, 3, 4]``, then those orbitals are not correlated in MP2 calculation.
"""

frac_occ = None
""" Fraction occupation number for occupied orbitals.

Should be list of floats.
"""

frac_vir = None
""" Fraction occupation number for virtual orbitals.

Should be list of floats.
"""

# endregion

# region Process control

incore_t_ijab = "auto"
""" Flag for tensor :math:`t_{ij}^{ab}` stored in memory or disk.

Parameters
----------
True
    Store tensor in memory.
False
    Store tensor in disk.
None
    Do not store tensor in either disk or memory.
"auto"
    Leave program to judge whether tensor locates.
(int)
    If tensor size exceeds this size (in MBytes), then store in disk. 
"""

integral_scheme = "ri"
""" Flag for MP2 integral.

Parameters
----------
"ri"
    Resolution of identity.
"conv"
    Conventional. Not recommended for large system.
"lt"
    Laplace transform with resolution of identity. Opposite-spin only.
"""

# endregion

# region Coefficients

coef_mp2 = 1
""" Coefficient of contribution to MP2 energy. """

coef_mp2_os = 1
""" Coefficient of opposite-spin contribution to MP2 energy. """

coef_mp2_ss = 1
""" Coefficient of same-spin contribution to MP2 energy. """

# endregion
