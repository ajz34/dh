from pyscf import lib, gto, dft, df
import numpy as np
from scipy.special import erfc

from pyscf.dh import util
from pyscf.dh.util import Params, HybridDict, XCType
from pyscf.dh.energy.rmp2 import driver_energy_rmp2
from pyscf.dh.energy.riepa import driver_energy_riepa
from pyscf.dh.energy.rdft import (
    kernel_energy_restricted_exactx, kernel_energy_restricted_noxc, kernel_energy_vv10,
    kernel_energy_purexc, get_rho)
from pyscf.dh.util import XCDH, XCList


class RDH(lib.StreamObject):
    """ Restricted doubly hybrid class. """
    _scf: dft.rks.RKS
    """ Self consistent object. """
    xc: XCDH
    """ Doubly hybrid functional exchange-correlation code. """
    params: Params
    """ Params object consisting of flags, tensors and results. """
    with_df: df.DF or None
    """ Density fitting object. """
    with_df_2: df.DF or None
    """ Density fitting object for customize ERI. Used in magnetic computation. """
    siepa_screen: callable
    """ Screen function for sIEPA. """

    def __init__(self, mf_or_mol, xc, params=None, with_df=None):
        # initialize parameters
        if params:
            self.params = params
        else:
            self.params = Params({}, HybridDict(), {})
        self.params.flags.set_default_dict(util.get_default_options())
        # set molecule
        mol = mf_or_mol if isinstance(mf_or_mol, gto.Mole) else mf_or_mol.mol
        # logger
        log = lib.logger.new_logger(verbose=mf_or_mol.verbose)
        self.verbose = mol.verbose
        self.log = lib.logger.new_logger(verbose=self.verbose)
        # generate mf object
        xc = xc if isinstance(xc, XCDH) else XCDH(xc)
        xc_code_scf = xc.xc_scf.token
        # generate with_df object
        auxbasis_ri = self.params.flags["auxbasis_ri"]
        if not isinstance(mf_or_mol, gto.Mole):
            # rks or uks
            mf = mf_or_mol
            integral_scheme_scf = self.params.flags["integral_scheme_scf"].replace("-", "").lower()
            if integral_scheme_scf.startswith("ri") and not hasattr(mf, "with_df"):
                log.warn("Option integral_scheme_scf is set to RI but no density-fitting object found in SCF instance!")
            if integral_scheme_scf in ("rijonx", "rij") and (not hasattr(mf, "only_dfj") or not mf.only_dfj):
                log.warn("Option integral_scheme_scf is set to RIJONX but actually not in SCF instance!")
            if auxbasis_ri is None and hasattr(mf, "with_df"):
                self.with_df = mf.with_df
                log.info("[INFO] Use density fitting object from SCF when evaluating post-SCF.")
            else:
                if auxbasis_ri is None:
                    auxbasis_ri = df.aug_etb(mol)
                    log.info("[INFO] Generate auxbasis_ri by aug-etb automatically.")
                self.with_df = df.DF(mol, auxbasis=auxbasis_ri)
        else:
            # no possible SCF with_df, then generate one
            if auxbasis_ri is None:
                auxbasis_ri = df.aug_etb(mol)
                log.info("[INFO] Generate auxbasis_ri by aug-etb automatically.")
            self.with_df = df.DF(mol, auxbasis=auxbasis_ri)
        # build SCF object
        if isinstance(mf_or_mol, gto.Mole):
            mol = mf_or_mol
            self.build_scf(mol, xc.xc_scf)
            mf = self._scf
        else:
            mf = mf_or_mol
            self._scf = mf
        # transform mf if pyscf.scf instead of pyscf.dft
        if not hasattr(mf, "xc"):
            log.warn("We only accept density functionals here.\n"
                     "If you pass an HF instance, we convert to KS object naively.")
            mf = mf.to_rks("HF") if self.restricted else mf.to_uks("HF")
            if mf.grids.weights is None:
                mf.initialize_grids()
        self._scf = mf
        # parse xc code
        if xc_code_scf != XCList(mf.xc, code_scf=True).token:
            log.warn("xc code for SCF functional is not the same from input and SCF object!\n" +
                     "Input xc for SCF: {:}\n".format(xc_code_scf) +
                     "SCF object xc   : {:}\n".format(self._scf.xc) +
                     "Input xc for eng: {:}\n".format(xc.xc_eng.token) +
                     "Use SCF object xc for SCF functional, input xc for eng as energy functional.")
            xc.xc_scf = XCList(mf.xc, code_scf=True)
        self.xc = xc
        # parse other objects
        self.with_df_2 = None
        self.siepa_screen = erfc

    def build_scf(self, mol, xc):
        """ Build self-consistent instance.

        This object handles situations of
        - specifing how density fitting performed;
        - self-defined numint object;
        - (possibly more ...)

        Several parameters used should be defined from parameter list:
        - integral_scheme_scf
        - auxbasis_jk

        Parameters
        ----------
        mol : gto.Mole
            Molecule instance.
        xc : str or XCList
            Full exchange-correlation for SCF evaluation.

        Returns
        -------
        dft.rks.RKS or dft.uks.UKS
        """
        log = self.log
        # build density fitting object
        if self.with_df._cderi is None:
            self.with_df.build()
        # build scf object anyway
        if isinstance(xc, str):
            xc_token = xc
        else:
            xc_token = xc.token
        if self.restricted:
            mf = dft.RKS(mol, xc=xc_token)
        else:
            mf = dft.UKS(mol, xc=xc_token)
        # test if numint should be customized
        ni = mf._numint
        try:
            ni._xc_type(xc_token)
        except KeyError:
            raise NotImplementedError("dft.numint.NumInt object should be customized!")
        # handle ri in scf
        integral_scheme_scf = self.params.flags["integral_scheme_scf"].replace("-", "").lower()
        is_ri_scf = integral_scheme_scf.startswith("ri")
        is_ri_jonx = integral_scheme_scf in ["rij", "rijonx"]
        if is_ri_scf:
            log.info("[INFO] SCF object uses RI.")
            log.info("[INFO] Do RI-J only without RI-K: {:}".format(is_ri_jonx))
            auxbasis_jk = self.params.flags["auxbasis_jk"]
            if auxbasis_jk is None:
                log.info("[INFO] Use auxiliary basis set from post-SCF configuration.")
                mf = mf.density_fit(df.aug_etb(mol), with_df=self.with_df, only_dfj=is_ri_jonx)
            else:
                mf = mf.density_fit(auxbasis=auxbasis_jk, only_dfj=is_ri_jonx)
        self._scf = mf

    def build(self):
        """ Build essential parts of doubly hybrid instance.

        Build process should be performed only once. Rebuild this instance shall not cost any time.
        """
        if not hasattr(self.scf, "xc"):
            self.log.warn("We only accept density functionals here.\n"
                     "If you pass an HF instance, we convert to KS object naively.")
            self._scf = self.scf.to_rks("HF") if self.restricted else self.scf.to_uks("HF")
        if self.scf.mo_coeff is None:
            self.log.info("[INFO] Molecular coefficients not found. Run SCF first.")
            self.scf.run()
        if self.scf.e_tot != 0 and not self.scf.converged:
            self.log.warn("SCF not converged!")
        if self.scf.grids.weights is None:
            self.scf.initialize_grids(dm=self.make_rdm1_scf())

    @property
    def scf(self) -> dft.rks.RKS:
        """ A more linting favourable replacement of attribute ``_scf``. """
        return self._scf

    @property
    def mol(self) -> gto.Mole:
        """ Molecular object. """
        if hasattr(self, "_scf") and self.scf:
            return self.scf.mol
        else:
            return self.with_df.mol

    @property
    def nao(self) -> int:
        """ Number of atomic orbitals. """
        return self.mol.nao

    @property
    def nmo(self) -> int:
        """ Number of molecular orbitals. """
        return self.scf.mo_coeff.shape[-1]

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
        return self.scf.mo_coeff[:, shuffle_frz]

    @property
    def mo_occ(self) -> np.ndarray:
        """ Molecular orbital occupation number. """
        shuffle_frz = self.get_shuffle_frz()
        return self.scf.mo_occ[shuffle_frz]

    @property
    def mo_energy(self) -> np.ndarray:
        """ Molecular orbital energy. """
        shuffle_frz = self.get_shuffle_frz()
        return self.scf.mo_energy[shuffle_frz]

    def make_rdm1_scf(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        dm = self.scf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
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
        return self.params.results["eng_dh_{:}".format(self.xc.xc_eng.token)]

    def get_Y_OV(self, regenerate=False) -> np.ndarray:
        """ Get cholesky decomposed ERI in MO basis (occ-vir part with frozen core).

        Dimension: (naux, nOcc, nVir)
        """
        nact, nOcc = self.nact, self.nOcc
        if regenerate or "Y_OV" not in self.params.tensors:
            self.log.info("[INFO] Generating `Y_OV` ...")
            Y_OV = util.get_cderi_mo(
                self.with_df, self.mo_coeff_act, None, (0, nOcc, nOcc, nact),
                self.mol.max_memory - lib.current_memory()[0])
            self.params.tensors["Y_OV"] = Y_OV
            self.log.info("[INFO] Generating `Y_OV` Done")
        else:
            Y_OV = self.params.tensors["Y_OV"]
        return Y_OV

    driver_energy_mp2 = driver_energy_rmp2
    driver_energy_iepa = driver_energy_riepa
    driver_energy_dh = driver_energy_dh
    kernel = driver_energy_dh

    kernel_energy_exactx = staticmethod(kernel_energy_restricted_exactx)
    kernel_energy_noxc = staticmethod(kernel_energy_restricted_noxc)
    kernel_energy_vv10 = staticmethod(kernel_energy_vv10)
    kernel_energy_purexc = staticmethod(kernel_energy_purexc)
    get_rho = staticmethod(get_rho)


def driver_energy_dh(mf_dh):
    """ Driver of multiple exchange-correlation energy component evaluation.

    Parameters
    ----------
    mf_dh : RDH
        Object of doubly hybrid (restricted).
    """
    xc = mf_dh.xc
    log = mf_dh.log
    mf_dh.build()
    ni = dft.numint.NumInt()
    result = dict()
    eng_tot = 0.
    # 0. noxc part
    result.update(mf_dh.kernel_energy_noxc(mf_dh.scf, mf_dh.make_rdm1_scf()))
    eng_tot += result["eng_noxc"]
    # 1. parse energy of xc_hyb
    # exact exchange contributions
    xc_exx = xc.xc_eng.extract_by_xctype(XCType.EXX)
    for info in xc_exx:
        log.info("[INFO] EXX to be evaluated: {:}".format(info.token))
        if info.name == "HF":
            result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf()))
            eng = result["eng_HF"]
            log.note("[RESULT] eng_HF {:20.12f}".format(eng))
            eng_tot += info.fac * eng
        else:
            assert info.name == "LR_HF"
            assert len(info.parameters) == 1
            omega = info.parameters[0]
            result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf(), omega))
            eng = result["eng_LR_HF({:})".format(omega)]
            log.note("[RESULT] eng_LR_HF({:}) {:20.12f}".format(omega, eng))
            eng_tot += info.fac * eng
    # general xc
    xc_general = xc.xc_eng.extract_by_xctype(XCType.RUNG_LOW).extract_by_xctype(lambda t: not (t & XCType.EXX))
    token = xc_general.token
    if len(token) > 0:
        # pure contribution
        log.info("DFT integral XC to be evaluated: {:}".format(token))
        grids = mf_dh.scf.grids
        rho = get_rho(mf_dh.mol, grids, mf_dh.make_rdm1_scf())
        result.update(mf_dh.kernel_energy_purexc([token], rho, grids.weights, mf_dh.restricted))
        eng = result["eng_purexc_{:}".format(token)]
        eng_tot += eng
        log.note("[RESULT] eng_purexc_{:} {:20.12f}".format(token, eng))
        # exchange contribution
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(token)
        if abs(hyb) > 1e-10:
            if "eng_HF" in result:
                eng = result["eng_HF"]
            else:
                result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf()))
                eng = result["eng_HF"]
                log.note("[RESULT] eng_HF {:20.12f}".format(eng))
            eng_tot += hyb * eng
        if abs(omega) > 1e-10 and abs(alpha - hyb) > 1e-10:
            if "eng_LR_HF({:})".format(omega) in result:
                eng = result["eng_LR_HF({:})".format(omega)]
            else:
                result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf(), omega))
                eng = result["eng_LR_HF({:})".format(omega)]
                log.note("[RESULT] eng_LR_HF({:}) {:20.12f}".format(omega, eng))
            eng_tot += (alpha - hyb) * eng

    # 2. other correlation
    # 2.1 IEPA (may also resolve evaluation of MP2)
    xc_iepa = xc.xc_eng.extract_by_xctype(XCType.IEPA)
    if len(xc_iepa) > 0:
        xc_iepa = xc.xc_eng.extract_by_xctype(XCType.IEPA | XCType.MP2)
        mf_dh.driver_energy_iepa()
        iepa_scheme = [info.name for info in xc_iepa]
        log.info("[INFO] Detected IEPAs: {:}".format(str(iepa_scheme)))
        with mf_dh.params.temporary_flags({"iepa_scheme": iepa_scheme}):
            mf_dh.driver_energy_iepa()
        for info in xc_iepa:
            eng = info.fac * (
                + info.parameters[0] * mf_dh.params.results["eng_{:}_OS".format(info.name)]
                + info.parameters[1] * mf_dh.params.results["eng_{:}_SS".format(info.name)])
            log.info("[RESULT] energy of {:} correlation: {:20.12f}".format(info.name, eng))
            eng_tot += eng
    # 2.2 MP2
    if len(xc_iepa) == 0:
        xc_mp2 = xc.xc_eng.extract_by_xctype(XCType.MP2 | XCType.RSMP2)
    else:
        xc_mp2 = xc.xc_eng.extract_by_xctype(XCType.RSMP2)
    if len(xc_mp2) > 0:
        log.info("[INFO] MP2 detected")
        # generate omega list
        # parameter of RSMP2: omega, c_os, c_ss
        omega_list = []
        for info in xc_mp2:
            if XCType.MP2 in info.type:
                omega_list.append(0)
            else:
                assert XCType.RSMP2 in info.type
                omega_list.append(info.parameters[0])
        assert len(set(omega_list)) == len(omega_list)
        # run mp2
        with mf_dh.params.temporary_flags({"omega_list_mp2": omega_list}):
            mf_dh.driver_energy_mp2()
        # parse results
        for info in xc_mp2:
            if XCType.MP2 in info.type:
                c_os, c_ss = info.parameters
                omega = 0
            else:
                omega, c_os, c_ss = info.parameters
            eng = info.fac * (
                + c_os * mf_dh.params.results[util.pad_omega("eng_MP2_OS", omega)]
                + c_ss * mf_dh.params.results[util.pad_omega("eng_MP2_SS", omega)])
            log.info("[RESULT] energy of {:} correlation: {:20.12f}".format(util.pad_omega("MP2", omega), eng))
            eng_tot += eng
    elif len(xc_mp2) > 1:
        raise ValueError("MP2 terms is larger than 1. Consider trim xc first.")
    # 2.3 VV10
    xc_vdw = xc.xc_eng.extract_by_xctype(XCType.VDW)
    for info in xc_vdw:
        # currently only implement VV10
        if info.name == "VV10":
            nlc_pars = info.parameters
            assert len(nlc_pars) == 2
            grids = mf_dh.scf.grids
            nlcgrids = mf_dh.scf.nlcgrids
            res = mf_dh.kernel_energy_vv10(
                mf_dh.mol, mf_dh.make_rdm1_scf(), nlc_pars, grids, nlcgrids,
                verbose=mf_dh.verbose)
            result.update(res)
            eng = info.fac * result["eng_VV10({:}; {:})".format(*nlc_pars)]
            log.info("[RESULT] energy of VV10({:}; {:}): {:20.12f}".format(*nlc_pars, eng))
            eng_tot += eng
        else:
            raise NotImplementedError("Currently VDW only accepts VV10!")
    # finalize
    result["eng_dh_{:}".format(xc.xc_eng.token)] = eng_tot
    log.note("[RESULT] Energy of xc {:}: {:20.12f}".format(xc.xc_eng.token, eng_tot))
    mf_dh.params.update_results(result)
    return mf_dh
