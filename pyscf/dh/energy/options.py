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

Parameters
----------
PySCF
    Rule from ``pyscf.data.elements.chemcore_atm``.
FreezeNobleGasCore
    Freeze largest noble gas core, which is default of G16 for non-6-31G-basis.
FreezeInnerNobleGasCore
    Freeze orbitals that next to largest noble gas core.
SmallCore
    Small frozen core from [1]_.
LargeCore
    Large frozen core from [1]_. This may also be the same to FreezeG2 in G16.

Warnings
--------
TODO: Function to parse frozen rule.

.. [1] Rassolov, Vitaly A, John A Pople, Paul C Redfern, and Larry A Curtiss. “The Definition of Core Electrons.”
       Chem. Phys. Lett. 350, (5–6), 573–76. https://doi.org/10.1016/S0009-2614(01)01345-8.
"""

frozen_list = None
""" Index list of frozen orbitals.

For example, if set to ``[0, 2, 3, 4]``, then those orbitals are not correlated in MP2 calculation.
"""

frac_num = None
""" Fraction occupation number list.

Should be list of floats, size as ``(nmo, )``.
"""

# endregion

# region Process control

incore_t_ijab = None
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

iepa_scheme = "MP2"
""" Flag for IEPA-like scheme.

List including the following schemes is also accepted.

Parameters
----------
"mp2"
    MP2 as basic method.
"IEPA"
    IEPA (independent electron pair approximation).
"sIEPA"
    Screened IEPA.
"DCPT2"
    DCPT2 (degeneracy-corrected second-order perturbation).
"MP2cr"
    MP2/cr (enhanced second-order treatment of electron pair).
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
