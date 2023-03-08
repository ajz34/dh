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
        self._round()

    def _round(self, ndigits=10):
        """ Round floats for XC factor and parameters. """
        self.fac = round(self.fac, ndigits)
        self.parameters = [round(f, ndigits) for f in self.parameters]

    @property
    def token(self) -> str:
        """ Standardlized name of XC contribution. """
        self._round()
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
    xc_list: List[XCInfo]
    """ List of detailed exchange and correlation information. """
    code_scf: bool
    """ Whether this exchange-correlation represents SCF functinoal. """

    def __init__(self):
        self.xc_list = []

    def build_from_token(self, token: str, code_scf: bool):
        self.xc_list = self.parse_token(token, code_scf)
        self.code_scf = code_scf
        return self

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
        match = re.findall(r"((,?)([+-]*)([0-9.]+\*)?(\w+)(\([0-9.,;]+\))?)", token)
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
            # try if parse_xc success
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
                    "Advanced component {:} should be in correlation contribution.\n"
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

    @property
    def token(self) -> str:
        """ Return a token that represents xc functional in a somehow standard way. """
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
        # 5. combine all lists
        token = " + ".join([info.token for info in xc_lists_sorted_x]).replace("+ -", "-")
        if len(xc_lists_sorted_c) > 0:
            token += ", " + " + ".join([info.token for info in xc_lists_sorted_c]).replace("+ -", "-")
        token = token.strip()
        # 6. sanity check
        rebuild = XCList().build_from_token(token, self.code_scf)
        if rebuild != self:
            warnings.warn("Returned token is not the same to the original xc list. Double check may required.")
        return token

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


if __name__ == '__main__':
    l = XCList().build_from_token("XYG3", False)
    print(l.xc_list)
