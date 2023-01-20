from pyscf import lib, gto, dft, df
import numpy as np
from scipy.special import erfc

from pyscf.dh import util
from pyscf.dh.util import Params, HybridDict
from pyscf.dh.energy.rmp2 import driver_energy_rmp2
from pyscf.dh.energy.riepa import driver_energy_riepa
from pyscf.dh.energy.rdft import (
    driver_energy_dh,
    kernel_energy_restricted_exactx, kernel_energy_restricted_noxc, kernel_energy_vv10,
    kernel_energy_purexc, get_rho)


class RDH(lib.StreamObject):
    """ Restricted doubly hybrid class. """
    mf: dft.rks.RKS
    """ Self consistent object. """
    xc: str or tuple
    """ Doubly hybrid functional exchange-correlation code. """
    params: Params
    """ Params object consisting of flags, tensors and results. """
    df_ri: df.DF or None
    """ Density fitting object. """
    df_ri_2: df.DF or None
    """ Density fitting object for customize ERI. Used in magnetic computation. """
    siepa_screen: callable
    """ Screen function for sIEPA. """

    def __init__(self, mf_or_mol, xc, params=None, df_ri=None):
        # initialize parameters
        if params:
            self.params = params
        else:
            self.params = Params(util.get_default_options(), HybridDict(), {})
        # generate mf object
        if isinstance(mf_or_mol, gto.Mole):
            mol = mf_or_mol
            if self.restricted:
                mf = dft.RKS(mol, xc=util.parse_dh_xc_code(xc, is_scf=True)[0]).density_fit(df.aug_etb(mol))
            else:
                mf = dft.UKS(mol, xc=util.parse_dh_xc_code(xc, is_scf=True)[0]).density_fit(df.aug_etb(mol))
        else:
            mf = mf_or_mol
        self.mf = mf
        log = lib.logger.new_logger(verbose=self.mol.verbose)
        # transform mf if pyscf.scf instead of pyscf.dft
        if self.mf.e_tot != 0 and not self.mf.converged:
            log.warn("SCF not converged!")
        if not hasattr(mf, "xc"):
            log.warn("We only accept density functionals here.\n"
                     "If you pass an HF instance, we convert to KS object naively.")
            self.mf = self.mf.to_rks("HF") if self.restricted else self.mf.to_uks("HF")
            if self.mf.grids.weights is None:
                self.mf.initialize_grids()
        # parse xc code
        if util.parse_dh_xc_code(xc, is_scf=True)[0].upper() != self.mf.xc.upper():
            log.warn("xc code for SCF functional is not the same from input and SCF object!\n" +
                     "Input xc for SCF: {:}\n".format(util.parse_dh_xc_code(xc, is_scf=True)[0]) +
                     "SCF object xc   : {:}\n".format(self.mf.xc) +
                     "Input xc for eng: {:}\n".format(xc if isinstance(xc, str) else xc[1]) +
                     "Use SCF object xc for SCF functional, input xc for eng as energy functional.")
            self.xc = self.mf.xc, (xc if isinstance(xc, str) else xc[1])
        else:
            self.xc = xc
        # parse density fitting object
        self.df_ri = df_ri
        if self.df_ri is None and hasattr(mf, "with_df"):
            self.df_ri = mf.with_df
        if self.df_ri is None:
            log.warn("Density-fitting object not found. "
                     "Generate a pyscf.df.DF object by default aug-etb settings.")
            self.df_ri = df.DF(self.mol, df.aug_etb(self.mol))
        # parse other objects

        self.df_ri_2 = None
        self.verbose = self.mol.verbose
        self.log = lib.logger.new_logger(verbose=self.verbose)
        self.siepa_screen = erfc

    def build(self):
        if self.mf.mo_coeff is None:
            self.mf.run()
        if self.mf.grids.weights is None:
            self.mf.initialize_grids(dm=self.make_rdm1_scf())

    @property
    def mol(self) -> gto.Mole:
        """ Molecular object. """
        return self.mf.mol

    @property
    def nao(self) -> int:
        """ Number of atomic orbitals. """
        return self.mol.nao

    @property
    def nmo(self) -> int:
        """ Number of molecular orbitals. """
        return self.mf.mo_coeff.shape[-1]

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
            self.params.tensors.create("mask_act", mask_act)
        return self.params.tensors["mask_act"]

    def get_shuffle_frz(self, regenerate=False) -> np.ndarray:
        """ Get shuffle indices array for frozen orbitals.

        For example, if orbitals 0, 2 are frozen, then this array will reshuffle
        MO orbitals to (0, 2, 1, 3, 4, ...).
        """
        if regenerate or "shuffle_frz" not in self.params.tensors:
            mask_act = self.get_mask_act(regenerate=regenerate)
            mo_idx = np.arange(self.nmo)
            nocc = self.nocc
            shuffle_frz_occ = np.concatenate([mo_idx[~mask_act][:nocc], mo_idx[mask_act][:nocc]])
            shuffle_frz_vir = np.concatenate([mo_idx[mask_act][nocc:], mo_idx[~mask_act][nocc:]])
            shuffle_frz = np.concatenate([shuffle_frz_occ, shuffle_frz_vir])
            self.params.tensors.create("shuffle_frz", shuffle_frz)
            if not np.allclose(shuffle_frz, np.arange(self.nmo)):
                self.log.warn("MO orbital indices will be shuffled.")
        return self.params.tensors["shuffle_frz"]

    @property
    def nCore(self) -> int:
        """ Number of frozen occupied orbitals. """
        mask_act = self.get_mask_act()
        return (~mask_act[:self.nocc]).sum()

    @property
    def nOcc(self) -> int:
        """ Number of active occupied orbitals. """
        mask_act = self.get_mask_act()
        return mask_act[:self.nocc].sum()

    @property
    def nVir(self) -> int:
        """ Number of active virtual orbitals. """
        mask_act = self.get_mask_act()
        return mask_act[self.nocc:].sum()

    @property
    def nFrzvir(self) -> int:
        """ Number of inactive virtual orbitals. """
        mask_act = self.get_mask_act()
        return (~mask_act[self.nocc:]).sum()

    @property
    def nact(self) -> int:
        """ Number of active molecular orbitals. """
        return self.get_mask_act().sum()

    def get_idx_frz_categories(self) -> tuple:
        """ Get indices of molecular orbital categories.

        This function returns 4 numbers:
        (nCore, nCore + nOcc, nCore + nOcc + nVir, nmo)
        """
        return self.nCore, self.nocc, self.nocc + self.nVir, self.nmo

    @property
    def mo_coeff(self) -> np.ndarray:
        """ Molecular orbital coefficient. """
        shuffle_frz = self.get_shuffle_frz()
        return self.mf.mo_coeff[:, shuffle_frz]

    @property
    def mo_occ(self) -> np.ndarray:
        """ Molecular orbital occupation number. """
        shuffle_frz = self.get_shuffle_frz()
        return self.mf.mo_occ[shuffle_frz]

    @property
    def mo_energy(self) -> np.ndarray:
        """ Molecular orbital energy. """
        shuffle_frz = self.get_shuffle_frz()
        return self.mf.mo_energy[shuffle_frz]

    def make_rdm1_scf(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        dm = self.mf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
        return dm

    @property
    def mo_coeff_act(self) -> np.ndarray:
        """ Molecular orbital coefficient (with frozen core). """
        return self.mo_coeff[:, self.nCore:self.nCore+self.nact]

    @property
    def mo_energy_act(self) -> np.ndarray:
        """ Molecular orbital energy (with frozen core). """
        return self.mo_energy[self.nCore:self.nCore+self.nact]

    @property
    def e_tot(self) -> float:
        """ Doubly hybrid total energy (obtained after running ``driver_energy_dh``). """
        return self.params.results["eng_dh_{:}".format(self.xc)]

    def get_Y_ov_act(self, regenerate=False) -> np.ndarray:
        """ Get cholesky decomposed ERI in MO basis (occ-vir part with frozen core).

        Dimension: (naux, nOcc, nVir)
        """
        nact, nOcc = self.nact, self.nOcc
        if regenerate or "Y_ov_act" not in self.params.tensors:
            self.log.info("[INFO] Generating `Y_ov_act` ...")
            Y_ov_act = util.get_cderi_mo(
                self.df_ri, self.mo_coeff_act, None, (0, nOcc, nOcc, nact),
                self.mol.max_memory - lib.current_memory()[0])
            self.params.tensors["Y_ov_act"] = Y_ov_act
            self.log.info("[INFO] Generating `Y_ov_act` Done")
        else:
            Y_ov_act = self.params.tensors["Y_ov_act"]
        return Y_ov_act

    driver_energy_mp2 = driver_energy_rmp2
    driver_energy_iepa = driver_energy_riepa
    driver_energy_dh = driver_energy_dh
    kernel = driver_energy_dh

    kernel_energy_exactx = staticmethod(kernel_energy_restricted_exactx)
    kernel_energy_noxc = staticmethod(kernel_energy_restricted_noxc)
    kernel_energy_vv10 = staticmethod(kernel_energy_vv10)
    kernel_energy_purexc = staticmethod(kernel_energy_purexc)
    get_rho = staticmethod(get_rho)
