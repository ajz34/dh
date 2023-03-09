"""
Exchange-correlation code parsing utility for doubly hybrid
"""
import itertools
import json
import os
import re
import copy
import warnings
from dataclasses import dataclass
from typing import List, Tuple
import enum
from enum import Flag
import numpy as np
from pyscf import dft


class XCType(Flag):
    """
    Exchange-correlation types.
    """
    HF = enum.auto()
    "Hartree-Fock"
    RSH = enum.auto()
    "Range-separate"
    EXX = HF | RSH
    "Exact exchange that requires ERI evaluation"

    LDA = enum.auto()
    "Local density approximation"
    GGA = enum.auto()
    "Generalized gradient approximation"
    MGGA = enum.auto()
    "meta-GGA"
    RUNG_LOW = EXX | LDA | GGA | MGGA
    "Low-rung (1st - 4th) approximation"

    MP2 = enum.auto()
    "MP2-like correlation contribution"
    IEPA = enum.auto()
    "IEPA-like correlation contribution"
    RPA = enum.auto()
    "RPA-like correlation contribution"
    RUNG_HIGH = MP2 | IEPA | RPA
    "High-rung (5th) approximation"

    VDW = enum.auto()
    "Van der Waals contribution"

    CORR = enum.auto()
    "Correlation contribution"
    EXCH = enum.auto()
    "Exchange contribution"
    HYB = enum.auto()
    "Hybrid DFA that includes multiple contributions of correlation and EXX contribution"
    PURE = enum.auto()
    "Pure low-rung DFA contribution that requires and only requires DFT integrand"

    UNKNOWN = 0


@dataclass
class XCInfo:
    """
    Exchange-correlation information.

    xc info refers to one component of exchange or correlation contribution.
    """

    fac: float
    """ Factor of xc contribution. """
    name: str
    """ Name of xc contribution. """
    parameters: List[float]
    """ Parameters list of xc contribution. """
    type: XCType
    """ Type of xc contribution. """

    def __init__(self, fac, name, parameters, typ):
        self.fac = fac
        self.name = name
        self.parameters = parameters
        self.type = typ
        self.round()

    def round(self, ndigits=10):
        """ Round floats for XC factor and parameters. """
        def adv_round(f, n):
            if f == round(f):
                return round(f)
            return round(f, n)

        self.fac = adv_round(self.fac, ndigits)
        self.parameters = [adv_round(f, ndigits) for f in self.parameters]

    @property
    def token(self) -> str:
        """ Standardlized name of XC contribution. """
        self.round()
        token = ""
        if self.fac < 0:
            token += "- "
        if abs(self.fac) != 1:
            token += str(abs(self.fac)) + "*"
        token += self.name
        if len(self.parameters) != 0:
            token += "(" + ";".join([str(f) for f in self.parameters]) + ")"
        token = token.upper()
        return token


class XCList:
    """
    Stores exchange and correlation information and processing xc tokens.
    """
    _xc_list: List[XCInfo]
    """ List of detailed exchange and correlation information. """
    code_scf: bool
    """ Whether this exchange-correlation represents SCF functinoal. """

    @property
    def xc_list(self):
        for info in self._xc_list:
            info.round()
        return self._xc_list

    @xc_list.setter
    def xc_list(self, xc_list):
        for info in xc_list:
            info.round()
        self._xc_list = xc_list

    def __init__(self, token=None, code_scf=None, **kwargs):
        self.xc_list = []
        if token:
            if not isinstance(code_scf, bool):
                raise ValueError("Must pass boolean option of `code_scf`.")
            self.build_from_token(token, code_scf, **kwargs)

    def build_from_token(
            self, token: str, code_scf: bool,
            merging=True,
    ):
        self.xc_list = self.parse_token(token, code_scf)
        self.code_scf = code_scf
        if merging:
            self.merging_exx()
            self.expand_by_spin()
            self.trim()
        return self

    # region XCList parsing

    @classmethod
    def parse_token(cls, token: str, code_scf: bool):
        """ Parse xc token to list of XCInfo without any modification or merging
        (except name with dash).

        Notes
        -----
        For regex of xc token, one may try: https://regex101.com/r/YeQU5m/1

        Groups of regex search

        0. match all token for an entire xc code for one term
        1. match if comma in string; if exists, then split exchange and correlation parts (pyscf convention);
        2. sign of factor
        3. factor (absolute value) with asterisk
        4. name of xc term
        5. list of parameters with parentheses
        """
        token = token.upper().replace(" ", "")
        for key, val in _NAME_WITH_DASH.items():
            token = token.replace(key, val)
        match = re.findall(r"((,?)([+-]*)([0-9.]+\*)?([\w@]+)(\([0-9.,;]+\))?)", token)
        # sanity check: matched patterns should be exactly equilvant to original token
        if token != "".join([group[0] for group in match]):
            raise ValueError(
                "XC token {:} is not successfully parsed.\n"
                "Regex match of this token becomes {:}".format(token, "".join([group[0] for group in match])))
        # check if splitting exchange and correlation is required
        # sanity check first: only zero or one comma
        comma_first = [group[0][0] == "," for group in match]
        split_xc = sum(comma_first)
        if split_xc > 1:
            raise ValueError("Only zero or one comma is required to split exchange and correlation parts!")
        index_split = -1
        if split_xc:
            index_split = comma_first.index(True)
        xc_list = []
        for n, re_group in enumerate(match):
            if not split_xc:
                guess_type = XCType.HYB
            elif n < index_split:
                guess_type = XCType.EXCH
            else:
                guess_type = XCType.CORR
            # see if is known doubly hybrid functional
            _, _, sgn, fac, name, parameters = re_group
            sgn = (-1)**sgn.count("-")
            fac = sgn if len(fac) == 0 else sgn * float(fac[:-1])
            if name in FUNCTIONALS_DICT and guess_type == XCType.HYB:
                assert len(parameters) == 0
                entry = FUNCTIONALS_DICT[name]
                if code_scf:
                    xc_list_add = cls.parse_token(entry.get("code_scf", entry["code"]), code_scf)
                else:
                    xc_list_add = cls.parse_token(entry["code"], code_scf)
                for xc_info in xc_list_add:
                    xc_info.fac *= fac
                xc_list += xc_list_add
                continue
            # otherwise, parse contribution information
            xc_info = cls.parse_xc_info(re_group, guess_type)
            if not (code_scf and not (xc_info.type & XCType.RUNG_LOW)):
                xc_list.append(xc_info)
        return xc_list

    @classmethod
    def parse_xc_info(cls, re_group: Tuple[str, ...], guess_type: XCType) -> XCInfo:
        """ Parse xc info from regex groups.

        See Also
        --------
        parse_token
        """
        _, _, sgn, fac, name, parameters = re_group
        # parse fac
        sgn = (-1)**sgn.count("-")
        if len(fac) > 0:
            fac = sgn * float(fac[:-1])
        else:
            fac = sgn
        # parse parameters
        if len(parameters) > 0:
            parameters = [float(f) for f in re.split(r"[,;]", parameters[1:-1])]
        else:
            parameters = []
        # parse xc type
        xc_info = XCInfo(fac, name, parameters, XCType.UNKNOWN)
        xc_type = cls.parse_xc_type(xc_info.name, guess_type)
        xc_info.type = xc_type
        return xc_info

    @classmethod
    def parse_xc_type(cls, name: str, guess_type: XCType) -> XCType:
        """ Try to parse xc type from name.

        This parsing utility generally uses PySCF, and do not check whether the exact type is.
        For example, "0.5*B88, 0.25*B88" will give the same result of "0.75*B88".
        Thus, for our parsing, we accept B88 as an correlation contribution currently.
        """
        # try libxc
        guess_name = name
        xc_type = XCType.UNKNOWN
        ni = dft.numint.NumInt()

        # detect simple cases of HF and RSH (RSH may have parameters, which may complicates)
        if name == "HF":
            return XCType.HF
        if name in ["LR_HF", "SR_HF", "RSH"]:
            return XCType.RSH

        # detect usual cases
        if guess_type & XCType.CORR:
            guess_name = "," + name
        try:
            # try if parse_xc success (must be low_rung)
            dft_type = ni._xc_type(guess_name)
            if guess_type != XCType.HYB:
                xc_type |= guess_type
            # parse dft type
            if dft_type == "NLC":
                raise KeyError("NLC code (with __VV10) is not accepted currently!")
            assert dft_type != "HF"
            DFT_TYPE_MAP = {
                "LDA": XCType.LDA,
                "GGA": XCType.GGA,
                "MGGA": XCType.MGGA,
            }
            xc_type |= DFT_TYPE_MAP[dft_type]
            # parse hf type
            rsh_and_hyb_coeff = list(ni.rsh_and_hybrid_coeff(guess_name))
            if rsh_and_hyb_coeff[1:3] != [0, 0]:
                xc_type |= XCType.HYB
            else:
                xc_type |= XCType.PURE
        except KeyError:
            # Key that is not parsible by pyscf must lies in high-rung or vdw contributions
            # as well as must be correlation contribution
            if not (guess_type & XCType.CORR):
                raise KeyError(
                    "Advanced component {:} is not low-rung exchange contribution.\n"
                    "Please consider add a comma to separate exch and corr.".format(name))
            if name in MP2_COMPONENTS:
                xc_type |= XCType.MP2 | XCType.CORR
            elif name in IEPA_COMPONENTS:
                xc_type |= XCType.IEPA | XCType.CORR
            elif name in RPA_COMPONENTS:
                xc_type |= XCType.RPA | XCType.CORR
            elif name in VDW_COMPONENTS:
                xc_type |= XCType.VDW | XCType.CORR
            else:
                raise KeyError("Unknown advanced C component {:}.".format(name))
        return xc_type

    # endregion

    # region trimming and merging

    def trim(self):
        """ Merge same items and remove terms that contribution coefficient (factor) is zero. """
        xc_list_trimed = []  # type: list[XCInfo]
        for info in self.xc_list:
            skip_info = False
            # see if mergable
            for info2 in xc_list_trimed:
                if info.name == info2.name and info.type == info2.type:
                    # addable parameters: add parameters
                    if info.name in _NAME_PARA_ADDABLE:
                        para = info.fac * np.array(info.parameters)
                        para2 = info2.fac * np.array(info2.parameters)
                        info2.fac = 1
                        info2.parameters = list(para + para2)
                        skip_info = True
                        break
                    # non-addable parameters: add factor
                    elif info.parameters == info2.parameters:
                        info2.fac += info.fac
                        skip_info = True
                        break
            if not skip_info:
                xc_list_trimed.append(info)
        # check values finally
        remove_index = []
        for n, info in enumerate(xc_list_trimed):
            info.round()
            if info.fac == 0:
                remove_index.append(n)
            if info.name in _NAME_PARA_ADDABLE and abs(np.array(info.parameters)).sum() == 0:
                remove_index.append(n)
        for n in remove_index[::-1]:
            xc_list_trimed.pop(n)
        self.xc_list = xc_list_trimed
        return self.sort()

    def expand_by_spin(self):
        """ Expand various PT2 methods by same-spin and oppo-spin coefficients.

        This only applies to several advanced correlations.
        """
        names_concern = _NAME_WITH_OS_SS
        names_concern_os = [s + "_OS" for s in names_concern]
        names_concern_ss = [s + "_SS" for s in names_concern]
        for info in self.xc_list:
            if info.name in names_concern:
                if len(info.parameters) == 0:
                    info.parameters = [info.fac, info.fac]
                    info.fac = 1
            elif info.name in names_concern_os:
                if len(info.parameters) != 0:
                    raise ValueError("{:} should not have parameters.".format(info.name))
                info.parameters = [info.fac, 0]
                info.name = info.name[:-3]
                info.fac = 1
            elif info.name in names_concern_ss:
                if len(info.parameters) != 0:
                    raise ValueError("{:} should not have parameters.".format(info.name))
                info.parameters = [0, info.fac]
                info.name = info.name[:-3]
                info.fac = 1
        return self.sort()

    def merging_exx(self):
        """ Merge HF and RSH contributions. """
        lst_exx = [info for info in self.xc_list if info.type & XCType.EXX]
        lst_other = [info for info in self.xc_list if not (info.type & XCType.EXX)]
        merged = {"HF": XCInfo(0, "HF", [], XCType.HF | XCType.EXCH)}
        for info in lst_exx:
            if info.name == "HF":
                merged["HF"].fac += info.fac
            elif info.name == "LR_HF":
                if len(info.parameters) != 1:
                    raise KeyError("LR_HF detected ({:}) but length of parameter is not 1.".format(info.token))
                omega = info.parameters[0]
                if omega not in merged:
                    merged[omega] = info
                else:
                    merged[omega].fac += info.fac
            elif info.name == "SR_HF":
                if len(info.parameters) != 1:
                    raise KeyError("SR_HF detected ({:}) but length of parameter is not 1.".format(info.token))
                omega = info.parameters[0]
                if omega not in merged:
                    merged[omega] = XCInfo(-info.fac, "LR_HF", [omega], info.type)
                    merged["HF"].fac += info.fac
                else:
                    merged[omega].fac -= info.fac
                    merged["HF"].fac += info.fac
            elif info.name == "RSH":
                if len(info.parameters) != 3:
                    raise KeyError("RHF detected ({:}) but length of parameter is not 3.".format(info.token))
                omega, alpha, beta = info.parameters
                if omega not in merged:
                    merged[omega] = XCInfo(-info.fac * beta, "LR_HF", [omega], info.type)
                    merged["HF"].fac += info.fac * (alpha + beta)
                else:
                    merged[omega].fac += - info.fac * beta
                    merged["HF"].fac += info.fac * (alpha + beta)
        lst = list(merged.values()) + lst_other
        self.xc_list = lst
        return self.sort()

    # endregion

    # region XCList basic utilities

    def copy(self):
        return copy.deepcopy(self)

    @property
    def token(self) -> str:
        """ Return a token that represents xc functional in a somehow standard way. """
        self.sort()
        xc_list_x = [info for info in self.xc_list if XCType.CORR not in info.type]
        xc_list_c = [info for info in self.xc_list if XCType.CORR in info.type]
        token = " + ".join([info.token for info in xc_list_x]).replace("+ -", "-")
        if len(xc_list_c) > 0:
            token += ", " + " + ".join([info.token for info in xc_list_c]).replace("+ -", "-")
        token = token.strip()
        return token

    def sort(self):
        """ Sort list of xc in unique way. """
        xc_list = self.xc_list.copy()

        # way of sort
        def token_for_sort(info: XCInfo):
            t = info.name
            t += str(info.parameters)
            t += str(info.type)
            t += str(info.fac)
            return t

        # 1. split general and correlation contribution
        xc_list_x = [info for info in xc_list if XCType.CORR not in info.type]
        xc_list_c = [info for info in xc_list if XCType.CORR in info.type]
        # 2. extract HF and RSH first, then pure and hybrid
        xc_lists_x = [
            [info for info in xc_list_x if xctype in info.type]
            for xctype in [XCType.HF, XCType.RSH, XCType.PURE, XCType.HYB]]
        # 3. extract pure, MP2,
        xc_lists_c = [
            [info for info in xc_list_c if xctype in info.type]
            for xctype in [XCType.PURE, XCType.MP2, XCType.IEPA, XCType.RPA, XCType.VDW]]
        # 4. sort each category
        for lst in xc_lists_x + xc_lists_c:
            lst.sort(key=token_for_sort)
        xc_lists_sorted_x = list(itertools.chain(*xc_lists_x))
        xc_lists_sorted_c = list(itertools.chain(*xc_lists_c))
        new_list = XCList()
        new_list.xc_list = xc_lists_sorted_x + xc_lists_sorted_c
        # sanity check
        if new_list != self:
            warnings.warn("Sorted xc_list is not the same to the original xc list. Double check may required.")
        self.xc_list = new_list.xc_list
        return self

    def __mul__(self, other: float):
        new_obj = copy.deepcopy(self)
        new_obj *= other
        return new_obj

    def __imul__(self, other: float):
        for xc_info in self.xc_list:
            xc_info.fac *= other
        return self

    def __eq__(self, other: "XCList"):
        tokens_self = [str(info) for info in self.xc_list]
        tokens_other = [str(info) for info in other.xc_list]
        tokens_self.sort()
        tokens_other.sort()
        return "".join(tokens_self) == "".join(tokens_other)

    # endregion


class XCDH:
    """ XC object for doubly hybrid functional.

    One may initialize XCDH object by one string (such as "XYG3"),
    or initialize by two strings indicating SCF and energy parts
    (such as ["PBE", "0.5*HF + 0.5*PBE, 0.75*PBE + 0.25*MP2"]).
    """
    xc_eng: XCList
    """ XC list for energy evaluation. """
    xc_scf: XCList
    """ XC list for SCF evaluation. """

    def __init__(self, token=None, **kwargs):
        if token:
            self.build_from_token(token, **kwargs)

    def build_from_token(self, token: str or tuple or list, **kwargs):
        if isinstance(token, (tuple, list)):
            if len(token) != 2:
                raise ValueError("If token passed in by tuple, then it must be two parts (scf, eng).")
            self.xc_scf = XCList().build_from_token(token[0], True, **kwargs)
            self.xc_eng = XCList().build_from_token(token[1], False, **kwargs)
        else:
            self.xc_scf = XCList().build_from_token(token, True, **kwargs)
            self.xc_eng = XCList().build_from_token(token, False, **kwargs)


# region modify when additional contribution added

# Advanced correlation contributors
MP2_COMPONENTS = [
    "MP2", "MP2_OS", "MP2_SS",
]

IEPA_COMPONENTS = [
    "MP2CR", "MP2CR_OS", "MP2CR_SS",
    "MP2CR2", "MP2CR2_OS", "MP2CR2_SS",
    "IEPA", "IEPA_OS", "IEPA_SS",
    "SIEPA", "SIEPA_OS", "SIEPA_SS",
]

RPA_COMPONENTS = [
    "DRPA",
]

VDW_COMPONENTS = [
    "VV10",
]

# xc names that have addable parameters
_NAME_PARA_ADDABLE = [
    "MP2", "MP2CR", "MP2CR2", "IEPA", "SIEPA",
]

# xc names that have suffix _OS and _SS
_NAME_WITH_OS_SS = [
    "MP2", "MP2CR", "MP2CR2", "IEPA", "SIEPA",
]

# endregion

# region parse automatically

# All 5-th functional detailed dictionary
FUNCTIONALS_DICT = dict()  # type: dict[str, dict]
dir_functionals = os.path.join(os.path.dirname(os.path.abspath(__file__)), "functionals")
for file_name in os.listdir(dir_functionals):
    with open(os.path.join(dir_functionals, file_name), "r") as f:
        FUNCTIONALS_DICT.update(json.load(f))

# Handle alias for 5-th functionals
FUNCTIONALS_DICT_ADD = dict()
for key in FUNCTIONALS_DICT:
    for alias in FUNCTIONALS_DICT[key].get("alias", []):
        FUNCTIONALS_DICT_ADD[alias] = FUNCTIONALS_DICT[key]
        FUNCTIONALS_DICT_ADD[alias]["see_also"] = key
FUNCTIONALS_DICT.update(FUNCTIONALS_DICT_ADD)

# handle underscores for 5-th functionals
FUNCTIONALS_DICT_ADD = dict()
for key in FUNCTIONALS_DICT:
    sub_key = re.sub("[-_/]", "", key)
    if sub_key != key:
        FUNCTIONALS_DICT_ADD[sub_key] = FUNCTIONALS_DICT[key]
        FUNCTIONALS_DICT_ADD[sub_key]["see_also"] = key
FUNCTIONALS_DICT.update(FUNCTIONALS_DICT_ADD)

# Dashed names
_NAME_WITH_DASH = {key.replace("_", "-"): key for key in FUNCTIONALS_DICT if "_" in key}
_NAME_WITH_DASH.update({
    key.replace("_", "-"): key
    for key in MP2_COMPONENTS + IEPA_COMPONENTS + RPA_COMPONENTS + VDW_COMPONENTS
    if "_" in key})
_NAME_WITH_DASH.update(dft.libxc._NAME_WITH_DASH)

# endregion


if __name__ == '__main__':
    l = XCList().build_from_token("XYG3", False)
    print(l.xc_list)
