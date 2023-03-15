"""
Information of one exchange-correlation term
"""

from dataclasses import dataclass
import re
from typing import Tuple
from .xctype import XCType
from .xcjson import ADV_CORR_ALIAS, ADV_CORR_DICT
from pyscf import dft


REGEX_XC = r"((,?)([+-]*)([0-9.]+\*)?([\w@]+)(\([0-9.,;-]+\))?)"


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
    parameters: list
    """ Parameters list of xc contribution. """
    type: XCType
    """ Type of xc contribution. """
    additional: dict
    """ Additional parameters.
    
    Parameters that is not sutiable to be passed as string token, or very advanced parameters.
    """

    def __init__(self, fac, name, parameters, typ):
        self.fac = fac
        self.name = name
        self.parameters = parameters
        self.type = typ
        self.additional = dict()
        self.round()

    def round(self, ndigits=10):
        """ Round floats for XC factor and parameters. """
        def adv_round(f, n):
            if f == round(f):
                return round(f)
            return round(f, n)

        self.fac = adv_round(self.fac, ndigits)
        self.parameters = [adv_round(f, ndigits) if isinstance(f, float) else str(f) for f in self.parameters]

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
            token += "(" + ", ".join([str(f) for f in self.parameters]) + ")"
        token = token.upper()
        return token

    @classmethod
    def parse_xc_info(cls, re_group: str or Tuple[str, ...], guess_type: XCType = XCType.UNKNOWN) -> "XCInfo":
        """ Parse xc info from regex groups.

        See Also
        --------
        parse_token
        """
        if isinstance(re_group, str):
            re_group = re_group.strip().replace(" ", "").upper()
            match = re.findall(REGEX_XC, re_group)
            if len(match) != 1:
                raise ValueError("Read from info failed in that prehaps multiple info is required.")
            if re_group != match[0][0]:
                raise ValueError(
                    "XC token {:} is not successfully parsed.\n"
                    "Regex match of this token becomes {:}".format(re_group, match[0][0]))
            re_group = match[0]
        _, comma, sgn, fac, name, parameters = re_group
        if comma == ",":
            guess_type = XCType.CORR
        if guess_type == XCType.UNKNOWN:
            guess_type = XCType.HYB
        assert guess_type in [XCType.HYB, XCType.EXCH, XCType.CORR]
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
        # build basic information
        xc_info = XCInfo(fac, name, parameters, XCType.UNKNOWN)
        # for advanced correlations, try to substitute alias first
        if xc_info.name in ADV_CORR_ALIAS:
            xc_info.name = ADV_CORR_ALIAS[xc_info.name]
            return cls.parse_xc_info(xc_info.token, guess_type)
        # parse xc type
        xc_type = cls.parse_xc_type(xc_info, guess_type)
        xc_info.type = xc_type
        # fill default parameters for advanced correlations if parameters not given
        if xc_info.name in ADV_CORR_DICT and len(xc_info.parameters) == 0:
            if "default_parameters" in ADV_CORR_DICT[xc_info.name]:
                xc_info.parameters = ADV_CORR_DICT[xc_info.name]["default_parameters"]
        # sanity check: for advanced correlations, number of parameter should be specified
        if xc_info.name in ADV_CORR_DICT:
            len_actual = len(xc_info.parameters)
            len_expected = len(ADV_CORR_DICT[xc_info.name]["addable"])
            if len_actual != len_expected:
                raise ValueError(
                    "Length of parameters of {:} should be {:} by design.".format(xc_info.token, len_expected))
        # handle special cases
        if xc_info.name == "SR_MP2":
            # in PySCF, short range omega is usually set to be smaller than zero
            xc_info.name = "RS_MP2"
            xc_info.parameters[0] = - xc_info.parameters[0]
        return xc_info

    @classmethod
    def parse_xc_type(cls, xc_info: "XCInfo", guess_type: XCType) -> XCType:
        """ Try to parse xc type from name.

        This parsing utility generally uses PySCF, and do not check whether the exact type is.
        For example, "0.5*B88, 0.25*B88" will give the same result of "0.75*B88".
        Thus, for our parsing, we accept B88 as an correlation contribution currently.
        """
        name = xc_info.name
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
            # except special cases
            if name == "VV10" and len(xc_info.parameters) > 0:
                raise KeyError("This VV10 is not GGA_XC_VV10, instead VV10 VDW with parameters.")
            # parse dft type
            if guess_type != XCType.HYB:
                xc_type |= guess_type
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
            # handle type
            # if not HYB | EXCH | CORR; then assign as HYB
            if not ((XCType.HYB | XCType.EXCH | XCType.CORR) & xc_type):
                xc_type |= XCType.HYB
            # if HYB, then CORR or EXCH will not exist
            if XCType.HYB in xc_type:
                xc_type &= ~(XCType.CORR | XCType.EXCH)
        except KeyError:
            # Key that is not parsible by pyscf must lies in high-rung or vdw contributions
            # as well as must be correlation contribution
            if not (guess_type & XCType.CORR):
                raise KeyError(
                    "Advanced component {:} is not low-rung exchange contribution.\n"
                    "Please consider add a comma to separate exch and corr.".format(name))
            type_map = {
                "MP2": XCType.MP2,
                "RSMP2": XCType.RSMP2,
                "IEPA": XCType.IEPA,
                "RPA": XCType.RPA,
                "VDW": XCType.VDW,
            }
            if name in ADV_CORR_DICT:
                xc_type |= XCType.CORR | type_map[ADV_CORR_DICT[name]["type"]]
            else:
                raise KeyError("Unknown advanced C component {:}.".format(name))
        return xc_type

