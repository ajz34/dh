import unittest
from pyscf.dh.util import XCList


class TestXCCode(unittest.TestCase):

    def test_xctype(self):

        def xc_io(inp, out, code_scf=False):
            x = XCList(inp, code_scf)
            self.assertEqual(x.token, out)

        # test usual functionals
        xc_io("B3LYPg", "B3LYPG")
        xc_io("PBE", "PBE")
        xc_io("PBE, PBE", "PBE, PBE")
        # test bdh
        xc_io("B2GPPLYP", "0.65*HF + 0.35*B88, 0.64*LYP + MP2(0.36, 0.36)")
        xc_io("B2GPPLYP", "0.65*HF + 0.35*B88, 0.64*LYP", code_scf=True)
        # test dh
        xc_io("XYG3", "0.8033*HF + 0.2107*B88 - 0.014*LDA, 0.6789*LYP + MP2(0.3211, 0.3211)")
        xc_io("ZRPS@PBE0", "0.5*HF + 0.5*PBE, 0.75*PBE + SIEPA(0.25, 0)")
        xc_io("ZRPS@PBE0", "PBE0", code_scf=True)
        # test advanced parameter addition
        xc_io(
            ", 0.75*MP2(2.5, 1.3) - 0.25*MP2_OS + 0.6*IEPA(2.3, 1.2) - 0.4*BGE2-OS",
            ", MP2(1.625, 0.975) + IEPA(0.98, 0.72)")
        xc_io(", 0.8*MP2(0.7, 0) - 0.56*MP2_OS", "")
        # test rsh
        xc_io(
            "0.6*HF - RSH(0.33, 0.2, 0.5) + 0.4*SR_HF(0.33) - 0.5*LR_HF(0.5)",
            "0.3*HF + 0.1*LR_HF(0.33) - 0.5*LR_HF(0.5),")
        xc_io(
            "0.6*HF - RSH(0.33, 0.2, 0.5) + 0.5*SR_HF(0.33) - 0.5*LR_HF(0.5)",
            "0.4*HF - 0.5*LR_HF(0.5),")
        # test VV10
        xc_io(
            "wB97M_V, VV10(6.0; 0.01) + 0.5*VV10(5.9; 0.0093)",
            "WB97M_V, 0.5*VV10(5.9, 0.0093) + VV10(6, 0.01)")
        # test rs-mp2
        xc_io(
            ", RS_MP2(0.5, 1, 0.5) + LR_MP2(0.5, 1, 0.8) + LR_MP2(0.3, 0.4, 1) + SR_MP2(0.3, 1, 1)",
            ", RS_MP2(-0.3, 1, 1) + RS_MP2(0.3, 0.4, 1) + RS_MP2(0.5, 2, 1.3)")
        # test name with hyphen
        xc_io(
            "M11-L + 0.25*M06-L - 2.0*CAM-B3LYP, PBE - 0.35*LDA - 0.25*MP2cr-OS + 1.5*MP2cr",
            "0.25*M06_L + M11_L - 2*CAM_B3LYP, - 0.35*LDA + PBE + MP2CR(1.25, 1.5)")
        xc_io("0.75*HF + 0.25*PBE,", "0.75*HF + 0.25*PBE,")

