import numpy as np
from typing import Tuple

from pyscf import gto, data, dft
import pyscf.data.elements


""" Accepted advanced correlation ingredient list. """
ACCEPTED_DH_CORR = {
    "mp2", "iepa", "siepa", "mp2cr", "mp2cr2"
}


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
        return tuple
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


def parse_dh_xc_token(token, is_corr):
    """ Parse functional token for doubly hybrid functional.

    Parameters
    ----------
    token : str
        Functional token. Should be something like ``0.5 * B88``.
    is_corr : bool
        Whether token represents exchange or correlation contribution.
    """
    token = token.strip()
    try:
        if is_corr:
            dft.libxc.parse_xc("," + token)
        else:
            dft.libxc.parse_xc(token)
        return False, token
    except KeyError:
        # parse number of token (code directly from pyscf.dft.libxc.parse_xc)
        token = token.lower()
        if token[0] == '-':
            sign = -1
            token = token[1:]
        else:
            sign = 1
        if '*' in token:
            fac, key = token.split('*')
            key = key.strip()
            if fac[0].isalpha():
                fac, key = key, fac
            fac = sign * float(fac)
        else:
            fac, key = sign, token
        # recognize "_os", "_ss"
        key = key.replace("-", "_").strip()
        if "_" not in key:
            key_name = key
            fac_os = fac_ss = fac
        else:
            key_split = key.split("_")
            if len(key_split) != 2:
                raise KeyError("Key {:} has more than 2 underscores and can't be recognized!".format(key))
            key_name, key_spin = key_split
            if key_spin == "os":
                fac_os, fac_ss = fac, 0
            elif key_spin == "ss":
                fac_os, fac_ss = 0, fac
            else:
                raise KeyError("Spin indicator {:} of key {:} is not recoginzed! Should be SS or OS."
                               .format(key_spin, key))
        if key_name not in ACCEPTED_DH_CORR:
            raise KeyError("{:} is not recognized as an doubly hybrid ingredient!".format(key_name))
        return True, (key_name, (fac_os, fac_ss))


def parse_dh_xc_code(xc_code, is_corr=False):
    """ Parse functional description for doubly hybrid functional.

    Rule of functional description (xc code) is similar to ``pyscf.dft.libxc.parse_xc``.

    In addition, advanced correlation (5th-rung correlation on Jacob's ladder) must be defined
    in correlation part of xc code, if exchange and correlation part of xc code are separated
    by comma.

    To specify oppo-spin and same-spin contributions, ``_OS`` and ``_SS`` should be added
    after the advanced correlation tokens.

    For example, energy evaluation functional of XYGJ-OS can be defined as follows:

    .. code::

        "0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP + 0.4364*MP2_OS"

    Result of parsed xc code is (not the same to result of ``pyscf.dft.libxc.parse_xc``)

    .. code::

        ('0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP', [('mp2', (0.4364, 0))])

    In the result tuple, first part is xc code of hybrid functional (<= 4th-rung), which
    should be able to be parsed by ``pyscf.dft.libxc.parse_xc``.

    Parameters
    ----------
    xc_code : str
        String representation of functional description.
    is_corr : bool
        (internal parameter) Changes token parse logic. This parameter is not designed
        for API users.

    Returns
    -------
    tuple[str, list[str, tuple[float, float]]]

    Notes
    -----
    Acceptable advanced correlation tokens are

    - MP2
    - IEPA
    - sIEPA
    - MP2cr
    - MP2cr2 (for restricted only)
    """
    # handle codes that exchange, correlation separated by comma
    if "," in xc_code:
        xc_code_x, xc_code_c = xc_code.split(",")
        xc_parsed_x = parse_dh_xc_code(xc_code_x, is_corr=False)
        xc_parsed_c = parse_dh_xc_code(xc_code_c, is_corr=True)
        if len(xc_parsed_x[1]) != 0:
            raise KeyError("Advanced correlation contribution should be defined in exchange part of xc code!")
        return xc_parsed_x[0] + ", " + xc_parsed_c[0], xc_parsed_c[1]
    # handle usual case
    tokens = xc_code.replace('-', '+-').replace(';+', ';').split('+')
    xc_hyb, xc_adv = [], []
    for token in tokens:
        is_adv, xc_info = parse_dh_xc_token(token, is_corr)
        if is_adv:
            xc_adv.append(xc_info)
        else:
            xc_hyb.append(xc_info)
    xc_hyb = " + ".join(xc_hyb)
    return xc_hyb, xc_adv
