from pyscf.dh import util
from pyscf.dh.energy.rdft import get_rho
from pyscf.dh.util import XCType


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
