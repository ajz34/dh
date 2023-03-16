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

auxbasis_ri = None
""" Auxiliary basis set for resolution of identity in post-SCF.

By default,
- If SCF object exists and is density-fitted, then default of post-SCF will still still use this instance;
- Otherwise, use aug-etb generated basis set.
"""

auxbasis_jk = None
""" Auxiliary basis set for resolution of identity in SCF.

By default,
- If SCF object does not exist, then use the same configuration of ``auxbasis_ri``;
- Otherwise, use aug-etb generated basis set.
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

incore_resp_eri = "auto"
""" Flag for tensor :math:`{}^\\mathrm{Exx} A_{ij}^{ab}` stored in memory or disk.

Exx here refers to exact exchange contribution in response tensor :math:`A_{ij}^{ab}`.
This includes HF contribution (for hybrids), or LR_HF contribution (for range-separates). 

Parameters
----------
True
    Store tensor in memory.
False
    Store tensor in disk.
"auto"
    Leave program to judge whether tensor locates.
(int)
    If tensor size exceeds this size (in MBytes), then store in disk.
"""

integral_scheme = "RI"
""" Flag for PT2 integral.

Parameters
----------
"RI"
    Resolution of identity.
"Conv"
    Conventional. Not recommended for large system.
"""

integral_scheme_scf = "RI"
""" Flag for SCF integral.

This extension does not focus on SCF, but will generate an SCF object if only molecule object passed in.
This flag only handles the case when molecule object passed.
If SCF object passed to evaluate doubly hybrids, then use everything provided from SCF object unless special cases.

Parameters
----------
"RI", "RI-JK"
    Resolution of identity for both coulomb and exchange (RI-JK).
"RI-J", "RIJONX"
    Resolution of identity for coulomb only (RI-JONX).
"Conv"
    Conventional.
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
    MP2/cr I (enhanced second-order treatment of electron pair).
"MP2cr2"
    MP2/cr II (not recommended, restricted only)
"""

omega_list_mp2 = [0]
""" Range-separate omega list of MP2.

Zero refers to no range-separate. Long/Short range uses posi/negative values.
"""

# endregion

# region Coefficients

coef_os = 1
""" Coefficient of opposite-spin contribution to MP2 energy. """

coef_ss = 1
""" Coefficient of same-spin contribution to MP2 energy. """

# endregion

# region Functional specific

ssr_x_fr = "LDA_X"
""" Full-range exchange functional for scaled short-range method. """

ssr_x_sr = "LDA_X_ERF"
""" Short-range exchange functional for scaled short-range method. """

ssr_c_fr = "LDA_C_PW"
""" Full-range correlation functional for scaled short-range method. """

ssr_c_sr = "LDA_C_PW_ERF"
""" Short-range correlation functional for scaled short-range method. """

# endregion
