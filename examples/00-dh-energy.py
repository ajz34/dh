# for development, import dh
# for general user, import pyscf.dh

from pyscf import gto, dft, mp
from dh import DFDH


def get_energy_xDH(mol: gto.Mole, xc_scf: str, c_pt2: float, xc_dh: str or None=None):
    # SCF part calculation
    mf_scf = dft.KS(mol, xc=xc_scf).run()
    e_nscf = mf_scf.e_tot
    # Non-consistent part calculation
    if xc_dh is not None:
        mf_nscf = dft.KS(mol, xc=xc_dh)
        e_nscf = mf_nscf.energy_tot(dm=mf_scf.make_rdm1())
    # PT2 contribution
    mf_pt2 = mp.MP2(mf_scf).run()
    e_pt2 = c_pt2 * mf_pt2.e_corr
    return e_nscf + e_pt2


if __name__ == '__main__':
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
    # XYG3     QChem   -76.2910536660
    #          PySCF   -76.2910535257
    #          RDFDH   -76.2910470614
    print(get_energy_xDH(mol, "B3LYPg", 0.3211, "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP"))
    print(DFDH(mol, xc="XYG3").run().e_tot)
    # B2PLYP   QChem   -76.29075997
    #          PySCF   -76.29075969
    #          RDFDH   -76.29076324
    print(get_energy_xDH(mol, "0.53*HF + 0.47*B88, 0.73*LYP", 0.27))
    print(DFDH(mol, xc="B2PLYP").run().e_tot)
    # XYGJ-OS  QChem   -76.1460262831
    #          RDFDH   -76.1460317123
    print(DFDH(mol, xc="XYGJ-OS").run().e_tot)
    # MP2      PySCF   -76.1108060780191
    #          RDFDH   -76.1107838767765
    print(mp.MP2(mol).run().e_tot)
    print(DFDH(mol, xc="MP2").run().e_tot)
    # Details:  QChem uses 99, 590 grid; i.e. XC_GRID 000099000590
    #                 no density fitting used in QChem for XYG3, B2PLYP, MP2
    #                 LT-SOS-RI-MP2 for XYGJ-OS
    #           PySCF default setting of dft.KS, no density fitting
    #           RDFDH default setting of dft.KS, density fitting basis from df.make_auxbasis
