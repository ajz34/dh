from pyscf import gto, dft, mp, df
from dh import RDFDH


def get_energy_xDH(mol: gto.Mole, xc_scf: str, c_pt2: float, xc_dh: str or None=None, auxbasis_jk=None, auxbasis_ri=None):
    # SCF part calculation
    mf_scf = dft.KS(mol, xc=xc_scf).density_fit(auxbasis=auxbasis_jk).run()
    e_nscf = mf_scf.e_tot
    # Non-consistent part calculation
    if xc_dh is not None:
        mf_nscf = dft.KS(mol, xc=xc_dh).density_fit(auxbasis=auxbasis_jk)
        e_nscf = mf_nscf.energy_tot(dm=mf_scf.make_rdm1())
    # PT2 contribution
    mf_pt2 = mp.MP2(mf_scf)
    mf_pt2.with_df = df.DF(mol, auxbasis=auxbasis_ri)
    mf_pt2.run()
    e_pt2 = c_pt2 * mf_pt2.e_corr
    return e_nscf + e_pt2


if __name__ == '__main__':
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
    # HF reference XYG3
    print(get_energy_xDH(mol, "HF", 0.3211, "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", "cc-pVDZ-jkfit", "cc-pVDZ-ri"))
    mf = RDFDH(mol, xc=["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]).run()
    print(mf.e_tot)
    mf = mf.nuc_grad_method().run()
    print(mf.grad_tot)
    mf = RDFDH(mol, xc=["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]).polar_method().run()
    print(mf.pol_tot)
    # B3LYP reference MP2
    print(get_energy_xDH(mol, "B3LYPg", 1, "HF", "cc-pVDZ-jkfit", "cc-pVDZ-ri"))
    mf = RDFDH(mol, xc=["B3LYPg", "HF", 1, 1, 1]).run()
    print(mf.e_tot)
    mf = mf.nuc_grad_method().run()
    print(mf.grad_tot)
    mf = RDFDH(mol, xc=["B3LYPg", "HF", 1, 1, 1]).polar_method().run()
    print(mf.pol_tot)
