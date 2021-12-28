from __future__ import annotations
from pyscf import dh
from pyscf.dh.dhutil import parse_xc_dh, gen_batch, calc_batch_size, HybridDict, timing, restricted_biorthogonalize
# typing import
from typing import Tuple, TYPE_CHECKING
from pyscf import lib
from pyscf.pbc import gto, scf, dft, df, mp
import numpy as np

@timing
def energy_elec_nc(mf: KDH, mo_coeff=None, h1e=None, vhf=None, **_):
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
            if mf.xc_n is None:  # if bDH-like functional, just return SCF energy
                return mf.mf_s.e_tot - mf.mf_s.energy_nuc(), None
        mo_coeff = mf.mf_s.mo_coeff
    mo_occ = mf.mf_s.mo_occ
    #if mo_occ is NotImplemented:
    #    if not mf.unrestricted:
    #        mo_occ = scf.hf.get_occ(mf.mf_s)
    #    else:
    #        mo_occ = scf.uhf.get_occ(mf.mf_s)
    #dm = mf.mf_s.make_rdm1(mo_coeff, mo_occ)
    dm = mf.mf_s.make_rdm1()
    dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    eng_nc = mf.mf_n.energy_elec(dm_kpts=dm, h1e_kpts=h1e, vhf=vhf)
    return eng_nc

@timing
def energy_elec_mp2(mf: KDH, mo_coeff=None, mo_energy=None, dfobj=None, 
                    #Y_ia_ri=None, t_ijab_blk=None, 
                    eval_ss=True, **_):
    if mo_coeff is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        if mf.mf_s.e_tot == 0:
            mf.run_scf()
        mo_energy = mf.mo_energy
    kmp = mp.KMP2(mf.mf_s, mo_coeff=mo_coeff)
    kmp.mo_energy = mo_energy
    eng = kmp.kernel()[0]
    return eng
    
# temporary approach since KMP2 cannot do SCS
@timing
def energy_elec_pt2(mf: KDH, params=None, eng_bi=None, **kwargs):
    if not mf.eval_pt2:  # not a PT2 functional
        return 0, 0, 0
    cc, c_os, c_ss = params if params else mf.cc, mf.c_os, mf.c_ss
    eng_bi = mf.energy_elec_mp2(eval_ss=mf.eval_ss, **kwargs)
    return (cc * eng_bi,  # Total
            0, 0)

class KDH(dh.rdfdh.RDFDH):

    def __init__(self,
                 mol: gto.Cell,
                 xc: str or tuple = "XYG3",
                 #auxbasis_jk: str or dict or None = None,
                 #auxbasis_ri: str or dict or None = None,
                 grids: dft.Grids = None,
                 #grids_cpks: dft.Grids = None,
                 unrestricted: bool = False,  # only for class initialization
                 kpts = np.zeros((1,3))
                 ):
        # tunable flags
        self.with_t_ijab = False  # only in energy calculation; polarizability is forced dump t2 to disk or mem
        self.max_memory = mol.max_memory
        # Parse xc code
        # It's tricky to say that, self.xc refers to SCF xc, and self.xc_dh refers to double hybrid xc
        # There should be three kinds of possible inputs:
        # 1) String: "XYG3"
        # 2) Tuple: ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1)
        # 3) Additional: (("0.69*HF + 0.31*PBE, 0.44*P86", None, 1, 0.52, 0.22), {"D3": ([0.48, 0, 0, 5.6, 0], 4)})
        self.xc_dh = xc
        if isinstance(xc, str):
            xc_list, xc_add = parse_xc_dh(xc)
        elif len(xc) == 5:  # here should assert xc is a tuple/list with 2 or 5 elements
            xc_list = xc
            xc_add = {}
        else:  # assert len(xc) == 2
            xc_list, xc_add = xc
        self.xc, self.xc_n, self.cc, self.c_os, self.c_ss = xc_list
        self.xc_add = xc_add
        # parse auxiliary basis
        #self.auxbasis_jk = auxbasis_jk = auxbasis_jk if auxbasis_jk else df.make_auxbasis(mol, mp2fit=False)
        #self.auxbasis_ri = auxbasis_ri = auxbasis_ri if auxbasis_ri else df.make_auxbasis(mol, mp2fit=True)
        #self.same_aux = True if auxbasis_jk == auxbasis_ri or auxbasis_ri is None else False
        # parse scf method
        self.unrestricted = unrestricted
        if unrestricted:
            mf_s = dft.KUKS(mol, kpts, xc=self.xc_n).rs_density_fit()
        else:
            mf_s = dft.KRKS(mol, kpts, xc=self.xc_n).rs_density_fit()
        self.grids = grids if grids else mf_s.grids                        # type: dft.grid.Grids
        #self.grids_cpks = grids_cpks if grids_cpks else self.grids         # type: dft.grid.Grids
        self.mf_s = mf_s                                                   # type: dft.rks.RKS
        self.mf_s.grids = self.grids
        # parse non-consistent method
        self.xc_n = None if self.xc_n == self.xc else self.xc_n            # type: str or None
        self.mf_n = self.mf_s                                              # type: dft.rks.RKS
        if self.xc_n:
            if unrestricted:
                self.mf_n = dft.KUKS(mol, kpts, xc=self.xc_n).rs_density_fit()
            else:
                self.mf_n = dft.KRKS(mol, kpts, xc=self.xc_n).rs_density_fit()
            self.mf_n.grids = self.mf_s.grids
            self.mf_n.grids = self.grids
        # parse hybrid coefficients
        self.ni = self.mf_s._numint
        self.cx = self.ni.hybrid_coeff(self.xc)
        self.cx_n = self.ni.hybrid_coeff(self.xc_n)
        # parse density fitting object
        #self.df_jk = mf_s.with_df  # type: df.DF
        #self.aux_jk = self.df_jk.auxmol
        #self.df_ri = df.DF(mol, auxbasis_ri) if not self.same_aux else self.df_jk
        #self.aux_ri = self.df_ri.auxmol
        # other preparation
        self.tensors = HybridDict()
        self.mol = mol
        self.nao = mol.nao  # type: int
        self.nocc = mol.nelec[0]
        # variables awaits to be build
        self.mo_coeff = NotImplemented
        self.mo_energy = NotImplemented
        self.mo_occ = NotImplemented
        #self.C = self.Co = self.Cv = NotImplemented
        #self.e = self.eo = self.ev = NotImplemented
        #self.D = NotImplemented
        self.nmo = self.nvir = NotImplemented
        #self.so = self.sv = self.sa = NotImplemented
        # results
        self.e_tot = NotImplemented
        self.eng_tot = self.eng_nc = self.eng_pt2 = self.eng_nuc = self.eng_os = self.eng_ss = NotImplemented
        # DANGEROUS PLACE
        # we could first initialize nmo as nao
        self.nmo = self.nao
        self.nvir = self.nmo - self.nocc
    
    @timing
    def build(self):
        pass

    @timing
    def run_scf(self, **kwargs):
        #self.mf_s.grids = self.mf_n.grids = self.grids
        #self.build()
        mf = self.mf_s
        if mf.e_tot == 0:
            mf.kernel(**kwargs)
        # prepare
        self.C = self.mo_coeff = mf.mo_coeff
        self.e = self.mo_energy = mf.mo_energy
        self.mo_occ = mf.mo_occ
        #self.D = mf.make_rdm1(mf.mo_coeff)
        nocc = self.nocc
        #nmo = self.nmo = self.C.shape[1]
        #self.nvir = nmo - nocc
        #self.so, self.sv, self.sa = slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
        #self.Co, self.Cv = self.C[:, self.so], self.C[:, self.sv]
        #self.eo, self.ev = self.e[self.so], self.e[self.sv]
        return self

    energy_elec_nc = energy_elec_nc
    energy_elec_mp2 = energy_elec_mp2
    energy_elec_pt2 = energy_elec_pt2

