import numpy as np
from typing import Tuple

from pyscf import gto, data, lib
import pyscf.data.elements


def parse_frozen_numbers(mol, rule=None) -> Tuple[int, int]:
    """ Parse frozen orbital numbers.

    Frozen orbitals are assumed to be paired. Returned frozen orbital numbers are actually half of core electrons.

    Parameters
    ----------
    mol : gto.Mole
        Molecular object.
    rule : str or tuple[int, int] or dict[str or int, tuple[int, int]] or None
        Rule to be parsed. This will be explained in detail in notes.

    Returns
    -------
    tuple[int, int]
        Tuple of (frozen occupied orbitals, frozen virtual orbitals).

    Notes
    -----
    This function only parse frozen numbers by element, instead of selecting proper orbitals to be frozen.

    Rule types:

    - If no rule given, then default is no frozen-orbitals..
    - As tuple[int, int]. First number is frozen (core) occupied orbitals; second number is frozen virtual
      orbitals.
    - As dict[str or int, tuple[int, int]]. Key of dict is element or charge of number; value of dict is the same to
      above.
    - As str, then it is should be the following rules:
        - ``PySCF``: Rule from ``pyscf.data.elements.chemcore_atm``.
        - ``FreezeNobleGasCore``: Freeze largest noble gas core, which is default of G16 for non-6-31G-basis.
        - ``FreezeInnerNobleGasCore``: Freeze orbitals that next to largest noble gas core.
        - ``SmallCore``: Small frozen core from [1]_.
        - ``LargeCore``: Large frozen core from [1]_. This may also be the same to FreezeG2 in G16.

    .. [1] Rassolov, Vitaly A, John A Pople, Paul C Redfern, and Larry A Curtiss. “The Definition of Core Electrons.”
           Chem. Phys. Lett. 350, (5–6), 573–76. https://doi.org/10.1016/S0009-2614(01)01345-8.
    """
    # Note that following tables are number of electrons, not orbitals, to be frozen.
    chemcore_atm = [
        0,  # ghost
        0,                                                                                   0,
        0,   0,                                                     2,   2,   2,   2,   2,   2,
        2,   2,                                                     10,  10,  10,  10,  10,  10,
        10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  18,  18,  18,  18,  18,  18,
        18,  18,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  36,  36,  36,  36,  36,  36,
        36,  36,
        # lanthanides
        36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  46,
                       46,  46,  46,  46,  46,  46,  46,  46,  46,  68,  68,  68,  68,  68,  68,
        68,  68,
        # actinides
        68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
                       100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 110, 110, 110, 110, 110]
    freeze_noble_gas_core = [
        0,  # ghost
        0,                                                                                   0,
        2,   2,                                                     2,   2,   2,   2,   2,   2,
        10,  10,                                                    10,  10,  10,  10,  10,  10,
        18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,
        36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,
        54,  54,
        # lanthanides
        54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,
                       54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,
        86,  86,
        # actinides
        86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,
                       86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86]
    freeze_inner_noble_gas_core = [
        0,  # ghost
        0,                                                                                   0,
        0,   0,                                                     0,   0,   0,   0,   0,   0,
        2,   2,                                                     2,   2,   2,   2,   2,   2,
        10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,
        18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,
        36,  36,
        # lanthanides
        36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,
                       36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,
        54,  54,
        # actinides
        54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,
                       54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54]
    freeze_small_core = [
        0,  # ghost
        0,                                                                                   0,
        2,   2,                                                     2,   2,   2,   2,   2,   2,
        2,   2,                                                     10,  10,  10,  10,  10,  10,
        10,  10,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,
        18,  18,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,
        36,  36,
        # lanthanides
        54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,
                       54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,
        54,  54,
        # actinides
        86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,
                       86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86]
    freeze_large_core = [
        0,  # ghost
        0,                                                                                   0,
        2,   2,                                                     2,   2,   2,   2,   2,   2,
        10,  10,                                                    10,  10,  10,  10,  10,  10,
        18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  28,  28,  28,  28,  28,  28,
        36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  36,  46,  46,  46,  46,  46,  46,
        54,  54,
        # lanthanides
        54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  68,
                       68,  68,  68,  68,  68,  68,  68,  68,  68,  78,  78,  78,  78,  78,  78,
        86,  86,
        # actinides
        86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86, 100,
                      100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 110, 110, 110, 110, 110]
    if rule is None:
        return 0, 0
    elif isinstance(rule, tuple):
        return rule
    elif isinstance(rule, dict):
        r = {}
        for key, val in rule.items():
            r[data.elements.charge(key)] = val
        f_occ, f_vir = 0, 0
        for i in range(mol.natm):
            if mol.atom_charge(i) in r:
                f_occ += r[mol.atom_charge(i)][0]
                f_vir += r[mol.atom_charge(i)][1]
        return f_occ, f_vir
    elif isinstance(rule, str):
        if rule.lower() == "pyscf":
            freeze_table = chemcore_atm
        elif rule.lower() == "FreezeNobleGasCore".lower():
            freeze_table = freeze_noble_gas_core
        elif rule.lower() == "FreezeInnerNobleGasCore".lower():
            freeze_table = freeze_inner_noble_gas_core
        elif rule.lower() == "SmallCore".lower():
            freeze_table = freeze_small_core
        elif rule.lower() == "LargeCore".lower():
            freeze_table = freeze_large_core
        else:
            raise ValueError("Freeze rule not recognized!")
        f_occ, f_vir = 0, 0
        for i in range(mol.natm):
            f_occ += freeze_table[mol.atom_charge(i)]
        assert f_occ % 2 == 0
        assert f_vir % 2 == 0
        return f_occ // 2, f_vir // 2
    else:
        raise ValueError("Type of Freeze rule is not recoginzed!")


def parse_frozen_list(mol, nmo=None, frz=None, rule=None):
    """ Parse frozen orbital list

    Parameters
    ----------
    mol : gto.Mole
        Molecular object.
    nmo : int or None
        Number of molecular orbitals. If not given, atomic orbital number will be filled.
    frz : list[int] or list[list[int]] or None
        List of orbitals to be freezed. May be restricted (list[int]) or unrestricted
        (two list[int] indicating frozen orbital indexes of alpha and beta orbitals).
    rule : str or tuple[int, int] or dict[str or int, tuple[int, int]] or None
        Rule to be parsed. This will be explained in detail in notes.

    Returns
    -------
    np.ndarray
        Mask of orbitals to be to be active.
    """
    if nmo is None:
        nmo = mol.nao
    if frz is None:
        # parse rules to give frozen orbitals
        if rule is None:
            # default, all orbitals are active
            return np.ones(nmo, dtype=bool)
        else:
            # give mask of active orbitals from parsed number of frozen occupied and virtual numbers
            f_occ, f_vir = parse_frozen_numbers(mol, rule)
            act = np.zeros(nmo, dtype=bool)
            act[np.array(range(f_occ, nmo - f_vir))] = True
            return act
    else:
        # parse list of frozen orbitals
        if not hasattr(frz, "__iter__"):
            raise ValueError("Variable `frz` must be list[int] or list[list[int]].")

        if len(frz) == 0:
            # empty frozen orbital case
            return np.ones(nmo, dtype=bool)

        if not hasattr(frz[0], "__iter__"):
            # one list of frozen orbitals:
            act = np.ones(nmo, dtype=bool)
            act[frz] = False
            return act
        else:
            # multiple lists of frozen orbitals
            act = np.ones((len(frz), nmo), dtype=bool)
            for i, f in enumerate(frz):
                act[i][f] = False
            return act


def restricted_biorthogonalize(t_ijab, cc, c_os, c_ss):
    """
    Biorthogonalize MP2 amplitude for restricted case.

    .. math::
        T_{ij}^{ab} = c_\\mathrm{c} \\big( c_\\mathrm{OS} t_{ij}^{ab} + c_\\mathrm{SS} (t_{ij}^{ab} - t_{ij}^{ba})
        \\big)

    Parameters
    ----------
    t_ijab : np.ndarray
        MP2 amplitude tensor.
    cc : float
        Coefficient of MP2 contribution.
    c_os : float
        Coefficient of MP2 opposite-spin contribution.
    c_ss : float
        Coefficient of MP2 same-spin contribution.

    Returns
    -------
    np.ndarray

    Notes
    -----
    Object of this function is simple. However, numpy's tensor transpose is notoriously slow.
    This function serves an API that can perform such kind of work in parallel efficiently.
    """
    # TODO: Efficiency may be further improved.
    coef_0 = cc * (c_os + c_ss)
    coef_1 = - cc * c_ss
    # handle different situations
    if abs(coef_1) < 1e-7:  # SS, do not make transpose
        return coef_0 * t_ijab
    else:
        t_shape = t_ijab.shape
        t_ijab = t_ijab.reshape((-1, t_ijab.shape[-2], t_ijab.shape[-1]))
        res = lib.transpose(t_ijab, axes=(0, 2, 1)).reshape(t_shape)
        t_ijab = t_ijab.reshape(t_shape)
        res *= coef_1
        res += coef_0 * t_ijab
        return res
