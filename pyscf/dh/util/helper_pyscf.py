import numpy as np
import re
from typing import Tuple

from pyscf import gto, data, dft, lib
import pyscf.data.elements


""" Accepted advanced correlation ingredient list. """
ACCEPTED_DH_CORR = {
    "MP2", "IEPA", "SIEPA", "MP2CR", "MP2CR2", "DCPT2"
}

""" Common name and detailed xc code of doubly hybrids. """
XC_DH_MAP = {   # [xc_for_scf (without advanced corr), xc_for_energy]
    "MP2": "HF + MP2",
    "XYG3": ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP + 0.3211*MP2"),
    "XYGJ_OS": ("B3LYPg", "0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP + 0.4364*MP2_OS"),
    "xDH_PBE0": ("PBE0", "0.8335*HF + 0.1665*PBE, 0.5292*PBE + 0.5428*MP2_OS"),
    "B2PLYP": "0.53*HF + 0.47*B88, 0.73*LYP + 0.27*MP2",
    "mPW2PLYP": "0.55*HF + 0.45*mPW91, 0.75*LYP + 0.25*MP2",
    "PBE0_DH": "0.5*HF + 0.5*PBE, 0.875*PBE + 0.125*MP2",
    "PBE_QIDH": "0.693361*HF + 0.306639*PBE, 0.666667*PBE + 0.333333*MP2",
    "PBE0_2": "0.793701*HF + 0.206299*PBE, 0.5*PBE + 0.5*MP2",
    "revXYG3": ("B3LYPg", "0.9196*HF - 0.0222*LDA + 0.1026*B88, 0.6059*LYP + 0.3941*MP2"),
    "XYG5": ("B3LYPg", "0.9150*HF + 0.0612*LDA + 0.0238*B88, 0.4957*LYP + 0.4548*MP2_OS + 0.2764*MP2_SS"),
    "XYG6": ("B3LYPg", "0.9105*HF + 0.1576*LDA - 0.0681*B88, 0.1800*VWN3 + 0.2244*LYP + 0.4695*MP2_OS + 0.2426*MP2_SS"),
    "XYG7": ("B3LYPg", "0.8971*HF + 0.2055*LDA - 0.1408*B88, 0.4056*VWN3 + 0.1159*LYP + 0.4502*MP2_OS + 0.2589*MP2_SS"),
    "revXYGJ_OS": ("B3LYPg", "0.8877*HF + 0.1123*LDA, -0.0697*VWN3 + 0.6167*LYP + 0.5485*MP2_OS"),
    "XYGJ_OS5": ("B3LYPg", "0.8928*HF + 0.3393*LDA - 0.2321*B88, 0.3268*VWN3 - 0.0635*LYP + 0.5574*MP2_OS"),
    "B2GPPLYP": "0.65*HF + 0.35*B88, 0.64*LYP + 0.36*MP2",
    "LS1DH_PBE": "0.75*HF + 0.25*PBE, 0.578125*PBE + 0.421875*MP2",
    "DSD_PBEP86_D3": "0.69*HF + 0.31*PBE, 0.44*P86 + 0.52*MP2_OS + 0.22*MP2_SS",
    "DSD_PBEPBE_D3": "0.68*HF + 0.32*PBE, 0.49*PBE + 0.55*MP2_OS + 0.13*MP2_SS",
    "DSD_BLYP_D3": "0.71*HF + 0.29*B88, 0.54*LYP + 0.47*MP2_OS + 0.40*MP2_SS",
    "DSD_PBEB95_D3": "0.66*HF + 0.34*PBE, 0.55*B95 + 0.46*MP2_OS + 0.09*MP2_SS",
    "B2PLYP_D3": "0.53*HF + 0.47*B88, 0.73*LYP + 0.27*MP2",
}

# upper xc dh common name without -, _, space
XC_DH_NORMED_SET = dict()
for _xc_code in XC_DH_MAP:
    XC_DH_NORMED_SET[re.sub("[-_ ]", "", _xc_code).upper()] = _xc_code


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


def standardize_token(token, is_corr=False):
    """ Standardize xc tokens.

    - hyphen changed to be underline
    - all characters are capitalized
    - no spaces unless factor is smaller than zero
    - parameter list are put into parentheses, separated by semicolon instead of comma
    - numeric factors lies before xc name
    - floats round up to 6th decimal

    A standardized token may be like ``0.5*B88``, ``- 0.3*VV10(6.0;0.01)``, ``- MP2(0.33;0.33)``, ``XYGJ_OS``,
    ``WB97X``.

    Parameters
    ----------
    token : str
        Input xc token.
    is_corr : bool
        Whether token represents correlation contribution.
        Only useful for DFT correlation, to be used in

    Returns
    -------
    dict
        Detailed decomposition of token.
    """
    token = token.strip().replace(" ", "").upper()
    # parse number of token (code directly from pyscf.dft.libxc.parse_xc)
    if token[0] == '-':
        sign = -1
        token = token[1:].strip()
    else:
        sign = 1
    assert token.count("*") <= 1
    if '*' in token:
        fac, key = token.strip().split('*')
        key = key.strip()
        if fac[0].isalpha():
            fac, key = key, fac
        fac = sign * float(fac)
    else:
        fac, key = sign, token.strip()
    fac = round(fac, 6)
    # parse parameter list
    if key.count("(") == 0:
        parameters = []
        name = key
    elif key.count("(") == 1:
        assert key[-1] == ")"
        name = key.split("(")[0]
        parameters = [round(float(f), 6) for f in key.split("(")[1][:-1].split(";")]
    else:
        assert False
    # recompose token string
    info = {
        "fac": fac,
        "name": name,
        "parameters": parameters}
    info = recompose_token(info)
    # check whether token is low_rung DFT xc
    try:
        if is_corr:
            dft.libxc.parse_xc("," + info["token"])
        else:
            dft.libxc.parse_xc(info["token"])
        is_low_rung = True
    except KeyError:
        is_corr = False
        is_low_rung = False
    info.update({
        "low_rung": is_low_rung,
        "corr": is_corr,
    })
    return info


def recompose_token(info):
    """ Re-compose a token by detailed information

    Parameters
    ----------
    info : dict

    Returns
    -------
    dict
        Dictionary of detailed information including token string.
    """
    info = info.copy()
    fac, name, parameters = info["fac"], info["name"], info["parameters"]
    # re-compose to a standardlized token
    re_token = name
    if np.allclose(abs(fac), 1):
        if fac < 0:
            re_token = "- " + re_token
    else:
        if fac < 0:
            re_token = "- " + str(abs(fac)) + "*" + re_token
        else:
            re_token = str(abs(fac)) + "*" + re_token
    if len(parameters) > 0:
        re_token = re_token + "(" + ";".join([str(f) for f in parameters]) + ")"
    info["token"] = re_token
    return info


def parse_dh_xc_code_detailed(xc_code):
    """ Parse detailed functional description for doubly hybrid functional.

    Rule of functional description (xc code) is similar to ``pyscf.dft.libxc.parse_xc``.

    To specify oppo-spin and same-spin contributions, ``_OS`` and ``_SS`` should be added
    after the advanced correlation tokens; or one can define oppo-spin and same-spin factors
    as parameters (``MP2(0.55;0.13)`` as an example of DSD-PBEPBE-D3).

    For example, energy evaluation functional of XYGJ-OS can be defined as follows:

    .. code::

        "0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP + 0.4364*MP2_OS"

    Result of parsed xc code is (not the same to result of ``pyscf.dft.libxc.parse_xc``)

    .. code::

        ('0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP',
         [detailed parsed tokens])

    In the result tuple, first part is xc code of hybrid functional (<= 4th-rung), which
    should be able to be parsed by ``pyscf.dft.libxc.parse_xc``.

    We use upper case to represent all xc codes after parsing.

    Parameters
    ----------
    xc_code : str
        String representation of functional detailed description.

    Returns
    -------
    list[dict]
        Parsed xc code of 3 parts: hybrid, advanced correlation, other.

    Notes
    -----
    Acceptable advanced correlation tokens are
    """
    if "," in xc_code:
        xc_code_x, xc_code_c = xc_code.split(",")
    else:
        xc_code_x, xc_code_c = xc_code, ""
    token_x = xc_code_x.strip().replace('-', '+-').replace(';+', ';').split('+')
    token_c = xc_code_c.strip().replace('-', '+-').replace(';+', ';').split('+')
    token_x = [t for t in token_x if t != ""]
    token_c = [t for t in token_c if t != ""]
    token_info = [standardize_token(t, False) for t in token_x] + [standardize_token(t, True) for t in token_c]
    return token_info


def extract_xc_code_low_rung(token_info):
    """ Obtain low rung part of xc code.

    Parameters
    ----------
    token_info : list[dict]

    Returns
    -------
    str
    """
    token_x_low_rung = [t["token"] for t in token_info if t["low_rung"] and not t["corr"]]
    token_c_low_rung = [t["token"] for t in token_info if t["low_rung"] and t["corr"]]
    # generate low rung tokens
    token_low_rung = " + ".join(token_x_low_rung).strip()
    if len(token_c_low_rung) > 0:
        token_low_rung += ", " + " + ".join(token_c_low_rung).strip()
    token_low_rung = token_low_rung.replace("+ - ", "- ")
    return token_low_rung


def handle_xc_code_pt2(token_info):
    """ Modify token info for PT2s.

    - Change something like ``MP2_OS`` to ``MP2(1;0)``
    - Add PT2 coefficients, such as ``0.5*MP2CR_OS + 0.2*MP2CR_SS`` to ``MP2CR(0.5;0.2)``

    Accepted PT2 codes:
    - MP2
    - IEPA
    - sIEPA
    - MP2cr
    - MP2cr2 (for restricted only)
    - DCPT2

    Parameters
    ----------
    token_info : list[dict]

    Returns
    -------
    list[dict]
    """
    accepted_pt2 = ["MP2", "IEPA", "SIEPA", "MP2CR", "MP2CR2", "DCPT2"]
    accepted_pt2_os = [s + "_OS" for s in accepted_pt2]
    accepted_pt2_ss = [s + "_SS" for s in accepted_pt2]
    token_info_ret = []
    token_info_pt2 = {}
    for info in token_info:
        if info["name"] in accepted_pt2 + accepted_pt2_os + accepted_pt2_ss:
            name = info["name"]
            parameters = info["parameters"]
            assert len(parameters) in [0, 2]
            if len(parameters) == 0:
                parameters = np.asarray([info["fac"], info["fac"]])
            else:
                parameters = info["fac"] * np.asarray(parameters)
            if name in accepted_pt2_os:
                name = name[:-3]
                parameters[1] = 0
            elif name in accepted_pt2_ss:
                name = name[:-3]
                parameters[0] = 0
            if name in token_info_pt2:
                updated_parameters = np.asarray(token_info_pt2[name]["parameters"])
                updated_parameters += parameters
                token_info_pt2[name]["parameters"] = list(np.round(updated_parameters, 6))
                token_info_pt2[name] = recompose_token(token_info_pt2[name])
            else:
                token_info_pt2[name] = {
                    "fac": 1,
                    "name": name,
                    "parameters": parameters,
                    "low_rung": False,
                    "corr": False,
                }
                token_info_pt2[name] = recompose_token(token_info_pt2[name])
        else:
            token_info_ret.append(info)
    token_info_ret += list(token_info_pt2.values())
    return token_info_ret


def parse_dh_xc_code(xc_code, is_scf):
    """ Parse functional description for doubly hybrid functional.

    Parameters
    ----------
    xc_code : str or tuple[str, str]
        String representation of functional description.
        Can be either detailed description or common name.
    is_scf : bool
        Self-consistent

    Returns
    -------
    list[dict]
        Parsed xc code of 3 parts: hybrid, advanced correlation, other.

    See Also
    --------
    parse_dh_xc_code_detailed
    """
    if isinstance(xc_code, tuple):
        xc_code = xc_code[0] if is_scf else xc_code[1]
    xc_code_normed = re.sub("[-_ ]", "", xc_code).upper()
    if xc_code_normed in XC_DH_NORMED_SET:
        xc_info = XC_DH_MAP[XC_DH_NORMED_SET[xc_code_normed]]
        if isinstance(xc_info, tuple):
            xc_code = xc_info[0] if is_scf else xc_info[1]
        else:
            xc_code = xc_info
    xc_parsed = parse_dh_xc_code_detailed(xc_code)
    if is_scf:
        return [info for info in xc_parsed if info["low_rung"]]
    else:
        return xc_parsed


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
