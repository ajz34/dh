from pyscf import lib, gto, dft, df
import numpy as np
from scipy.special import erfc

from pyscf.dh import util
from pyscf.dh.util import Params, HybridDict
from pyscf.dh.energy.rmp2 import driver_energy_rmp2
from pyscf.dh.energy.riepa import driver_energy_riepa
from pyscf.dh.energy.rdft import driver_energy_dh


class RDH(lib.StreamObject):
    """ Restricted doubly hybrid class. """
    mf: dft.rks.RKS
    """ Self consistent object. """
    xc_dh: str
    """ Exchange-correlation code for energy evaluation of doubly hybrid functional. """
    params: Params
    """ Params object consisting of flags, tensors and results. """
    df_ri: df.DF or None
    """ Density fitting object. """
    df_ri_2: df.DF or None
    """ Density fitting object for customize ERI. Used in magnetic computation. """
    siepa_screen: callable
    """ Screen function for sIEPA. """

    def __init__(self, mf_or_xc=NotImplemented, xc_dh=None, params=None, df_ri=None):
        if isinstance(mf_or_xc, str):
            raise NotImplementedError
        else:
            mf = mf_or_xc

        self.mf = mf
        log = lib.logger.new_logger(verbose=self.mol.verbose)
        if not self.mf.converged:
            log.warn("SCF not converged!")
        if not hasattr(mf, "xc"):
            log.warn("We only accept density functionals here.\n"
                     "If you pass an HF instance, we convert to KS object naively.")
            self.mf = self.mf.to_rks("HF") if self.restricted else self.mf.to_uks("HF")
            self.mf.initialize_grids(self.mol, self.mf.make_rdm1())
        self.df_ri = df_ri
        if self.df_ri is None and hasattr(mf, "with_df"):
            self.df_ri = mf.with_df
        if self.df_ri is None:
            log.warn("Density-fitting object not found. "
                     "Generate a pyscf.df.DF object by default aug-etb settings.")
            self.df_ri = df.DF(self.mol, df.aug_etb(self.mol))

        if params:
            self.params = params
        else:
            self.params = Params(util.get_default_options(), HybridDict(), {})

        self.df_ri_2 = None
        self.verbose = self.mol.verbose
        self.log = lib.logger.new_logger(verbose=self.verbose)
        self.siepa_screen = erfc

    @property
    def mol(self) -> gto.Mole:
        """ Molecular object. """
        return self.mf.mol

    @property
    def mo_coeff(self) -> np.ndarray:
        """ Molecular orbital coefficient. """
        return self.mf.mo_coeff

    @property
    def mo_occ(self) -> np.ndarray:
        """ Molecular orbital occupation number. """
        return self.mf.mo_occ

    @property
    def mo_energy(self) -> np.ndarray:
        """ Molecular orbital energy. """
        return self.mf.mo_energy

    @property
    def nao(self) -> int:
        """ Number of atomic orbitals. """
        return self.mol.nao

    @property
    def nmo(self) -> int:
        """ Number of molecular orbitals. """
        return self.mo_coeff.shape[-1]

    @property
    def nocc(self) -> int:
        """ Number of occupied molecular orbitals. """
        return self.mol.nelec[0]

    @property
    def nvir(self) -> int:
        """ Number of virtual molecular orbitals. """
        return self.nmo - self.nocc

    @property
    def restricted(self) -> bool:
        """ Restricted closed-shell or unrestricted open-shell. """
        return True

    def get_mask_act(self, regenerate=False) -> np.ndarray:
        """ Get mask of active orbitals.

        Dimension: (nmo, ), boolean array
        """
        if regenerate or "mask_act" not in self.params.tensors:
            frozen_rule = self.params.flags["frozen_rule"]
            frozen_list = self.params.flags["frozen_list"]
            mask_act = util.parse_frozen_list(self.mol, self.nmo, frozen_list, frozen_rule)
            self.params.tensors["mask_act"] = mask_act
        return self.params.tensors["mask_act"]

    @property
    def nmo_f(self) -> int:
        """ Number of molecular orbitals (with frozen core). """
        return self.get_mask_act().sum()

    @property
    def nocc_f(self) -> int:
        """ Number of occupied orbitals (with frozen core). """
        return self.get_mask_act()[:self.nocc].sum()

    @property
    def nvir_f(self) -> int:
        """ Number of virtual orbitals (with frozen core). """
        return self.nmo_f - self.nocc_f

    @property
    def mo_coeff_f(self) -> np.ndarray:
        """ Molecular orbital coefficient (with frozen core). """
        return self.mo_coeff[:, self.get_mask_act()]

    @property
    def mo_energy_f(self) -> np.ndarray:
        """ Molecular orbital energy (with frozen core). """
        return self.mo_energy[self.get_mask_act()]

    def get_Y_ov_f(self, regenerate=False) -> np.ndarray:
        """ Get cholesky decomposed ERI in MO basis (occ-vir part with frozen core).

        Dimension: (naux, nocc_f, nvir_f)
        """
        nmo_f, nocc_f = self.nmo_f, self.nocc_f
        if regenerate or "Y_ov_f" not in self.params.tensors:
            self.log.log("[INFO] Generating `Y_ov_f` ...")
            Y_ov_f = util.get_cderi_mo(self.df_ri, self.mo_coeff_f, None, (0, nocc_f, nocc_f, nmo_f),
                                       self.mol.max_memory - lib.current_memory()[0])
            self.params.tensors["Y_ov_f"] = Y_ov_f
            self.log.log("[INFO] Generating `Y_ov_f` Done")
        else:
            Y_ov_f = self.params.tensors["Y_ov_f"]
        return Y_ov_f

    driver_energy_mp2 = driver_energy_rmp2
    driver_energy_iepa = driver_energy_riepa
    driver_energy_dh = driver_energy_dh
