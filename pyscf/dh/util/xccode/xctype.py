import enum
from enum import Flag


class XCType(Flag):
    """
    Exchange-correlation types.
    """
    # region exact exchange
    HF = enum.auto()
    "Hartree-Fock"
    RSH = enum.auto()
    "Range-separate"
    EXX = HF | RSH
    "Exact exchange that requires ERI evaluation"
    # endregion

    # region low rung category
    LDA = enum.auto()
    "Local density approximation"
    GGA = enum.auto()
    "Generalized gradient approximation"
    MGGA = enum.auto()
    "meta generalized gradient approximation"
    RUNG_LOW = EXX | LDA | GGA | MGGA
    "Low-rung (1st - 4th) approximation"
    # endregion

    # region high rung
    MP2 = enum.auto()
    "MP2 correlation contribution"
    RSMP2 = enum.auto()
    "Range-separate MP2 correlation contribution"
    IEPA = enum.auto()
    "IEPA-like correlation contribution"
    RPA = enum.auto()
    "RPA-like correlation contribution"
    RUNG_HIGH = MP2 | RSMP2 | IEPA | RPA
    "High-rung (5th) approximation"
    # endregion

    # region vDW
    VV10 = enum.auto()
    "Vydrov and van Voorhis 2010"
    VDW = VV10
    "Van der Waals contribution"
    # endregion

    # region type of exch, corr or hyb
    EXCH = enum.auto()
    "Exchange contribution"
    CORR = enum.auto()
    "Correlation contribution"
    HYB = enum.auto()
    "Hybrid DFA that includes multiple contributions of correlation and EXX contribution"
    # endregion

    # region special configurations
    SSR = enum.auto()
    "Short separate-range"
    # endregion

    UNKNOWN = 0

    def check_sanity(self):
        """ Check contradictory flags. """

        # check contradictory
        def contradictory(*args):
            exist = [self & arg != self.UNKNOWN for arg in args]
            assert sum(exist) <= 1

        # decompose flag
        def decompose(flag):
            bits = bin(flag.value)[2:][::-1]
            return [self.__class__(2**n) for n, v in enumerate(bits) if v == "1"]

        contradictory(*decompose(self.EXX))
        contradictory(*decompose(self.RUNG_LOW))
        contradictory(*decompose(self.RUNG_HIGH))
        contradictory(*decompose(self.VDW))
        contradictory(self.CORR, self.EXCH, self.HYB)

        # check special rules
        # SSR
        if self.SSR in self:
            assert self.EXCH in self or self.CORR in self
            assert self & self.RUNG_LOW

        # additional check for parameters list
        contradictory(*self.def_parameters().keys())

    @classmethod
    def def_parameters(cls):
        """ Definition dictionary of addability of parameter list for XCInfo.

        If some xc type is not listed, then this type should not have a parameter.
        """
        dct = {
            cls.RSH: [
                ["range-separate omega", False],
            ],
            cls.MP2: [
                ["oppo-spin coefficient", True],
                ["same-spin coefficient", True],
            ],
            cls.RSMP2: [
                ["range-separate omega", False],
                ["oppo-spin coefficient", True],
                ["same-spin coefficient", True],
            ],
            cls.IEPA: [
                ["oppo-spin coefficient", True],
                ["same-spin coefficient", True],
            ],
            cls.VV10: [
                ["parameter that controls damping of R6", False],
                ["parameter for C6 coefficient", False],
            ],
            cls.SSR: [
                ["functional to be scaled by range-separate coefficient", False],
                ["range-separate omega", False],
            ]
        }
        return dct

    def addable_parameters(self):
        """ List of addability of parameters """
        if self not in self.def_parameters():
            return False
        return [lst[1] for lst in self.def_parameters()[self]]
