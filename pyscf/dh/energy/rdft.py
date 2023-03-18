import typing
from types import MethodType

from pyscf.dh import util
from pyscf import dft, lib
import numpy as np
from pyscf.dh.util import DictWithDefault, XCList, XCType

if typing.TYPE_CHECKING:
    pass


def kernel_energy_restricted_exactx(mf, dm, omega=None):
    """ Evaluate exact exchange energy (for either HF and long-range).

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_k`` member function.
    dm : np.ndarray
        Density matrix.
    omega : float or None
        Parameter of long-range ERI integral :math:`\\mathrm{erfc} (\\omega r_{12}) / r_{12}`.
    """
    hermi = 1 if np.allclose(dm, dm.T.conj()) else 0
    vk = mf.get_k(dm=dm, hermi=hermi, omega=omega)
    ex = - 0.25 * np.einsum('ij, ji ->', dm, vk)
    ex = util.check_real(ex)
    # results
    result = dict()
    if omega is None:
        result["eng_HF"] = ex
    else:
        result["eng_LR_HF({:})".format(omega)] = ex
    return result


def kernel_energy_restricted_noxc(mf, dm):
    """ Evaluate energy contributions that is not exchange-correlation.

    Note that some contributions (such as vdw) is not considered.

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_hcore``, ``get_j`` member functions.
    dm : np.ndarray
        Density matrix.
    """
    hermi = 1 if np.allclose(dm, dm.T.conj()) else 0
    hcore = mf.get_hcore()
    vj = mf.get_j(dm=dm, hermi=hermi)
    eng_nuc = mf.mol.energy_nuc()
    eng_hcore = np.einsum('ij, ji ->', dm, hcore)
    eng_j = 0.5 * np.einsum('ij, ji ->', dm, vj)
    eng_hcore = util.check_real(eng_hcore)
    eng_j = util.check_real(eng_j)
    eng_noxc = eng_hcore + eng_nuc + eng_j
    # results
    results = dict()
    results["eng_nuc"] = eng_nuc
    results["eng_hcore"] = eng_hcore
    results["eng_j"] = eng_j
    results["eng_noxc"] = eng_noxc
    return results


def get_rho(mol, grids, dm):
    """ Obtain density on DFT grids.

    Note that this function always give density ready for meta-GGA.
    Thus, returned density grid dimension is (6, ngrid).
    For more details, see docstring in ``pyscf.dft.numint.eval_rho``.

    This function accepts either restricted or unrestricted density matrices.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    grids : dft.grid.Grids
        DFT grids object. Dimension (nao, nao) or (nset, nao, nao).
    dm : np.ndarray
        Density matrix.

    Returns
    -------
    np.ndarray
        Density grid of dimension (6, ngrid) or (nset, 6, ngrid).
    """
    ngrid = grids.weights.size
    ni = dft.numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm)
    rho = np.empty((nset, 6, ngrid))
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, deriv=2, max_memory=2000):
        p0, p1 = p1, p1 + weight.size
        for idm in range(nset):
            rho[idm, :, p0:p1] = make_rho(idm, ao, mask, 'MGGA')
    if nset == 1:
        rho = rho[0]
    return rho


def kernel_energy_purexc(xc_list, rho, weights, restricted, numint=None):
    """ Evaluate energy contributions of exchange-correlation effects.

    Note that this kernel does not count HF, LR_HF and advanced correlation into account.
    To evaluate exact exchange (HF or LR_HF), use ``kernel_energy_restricted_exactx``.

    Parameters
    ----------
    xc_list : list
        List of xc codes.
    rho : np.ndarray
        Full list of density grids. Dimension (>1, ngrid) or (nset, >1, ngrid).
    weights : np.ndarray
        Full list of DFT grid weights.
    restricted : bool
        Indicator of restricted or unrestricted of incoming rho.
    numint : dft.numint.NumInt
        Special numint item if required.

    See Also
    --------
    kernel_energy_restricted_exactx
    """
    if isinstance(xc_list, str):
        xc_list = [xc_list]
    results = {}
    ni = dft.numint.NumInt() if numint is None else numint
    if restricted:
        wrho0 = rho[0] * weights
    else:
        wrho0 = rho[:, 0].sum(axis=0) * weights

    for xc in xc_list:
        exc = ni.eval_xc_eff(xc, rho, deriv=0)[0]
        results["eng_purexc_" + xc] = exc @ wrho0
    return results


def kernel_energy_vv10(mol, dm, nlc_pars, grids=None, nlcgrids=None, verbose=lib.logger.NOTE):
    log = lib.logger.new_logger(verbose=verbose)
    if grids is None:
        log.warn("VV10 grids not found. Use default grids of PySCF for VV10.")
        grids = dft.Grids(mol).build()
    rho = get_rho(mol, grids, dm)
    if nlcgrids is None:
        nlcgrids = grids
        vvrho = rho
    else:
        nlcgrids.build()
        vvrho = get_rho(mol, nlcgrids, dm)
    # handle unrestricted case
    if len(rho.shape) == 3:
        rho = rho[0] + rho[1]
        vvrho = vvrho[0] + vvrho[1]
    exc_vv10, _ = dft.numint._vv10nlc(rho, grids.coords, vvrho, nlcgrids.weights, nlcgrids.coords, nlc_pars)
    eng_vv10 = (rho[0] * grids.weights * exc_vv10).sum()
    result = dict()
    result["eng_VV10({:}; {:})".format(*nlc_pars)] = eng_vv10
    return result


def numint_customized(flags, xc):
    """ Customized (specialized) numint for a certain given xc.

    Parameters
    ----------
    flags : DictWithDefault
        Flags (customizable parameters) for this computation.
    xc : XCList
        xc list evaluation. Currently only accept low-rung (without VV10) and scaled short-range functionals.

    Returns
    -------
    dft.numint.NumInt
        A customized numint object that only evaluates dft grids on given xc.
    """
    # extract the functionals that is parsable by PySCF
    ni_custom = dft.numint.NumInt()  # customized numint, to be returned
    ni_original = dft.numint.NumInt()  # original numint
    xc_parsable = xc.extract_by_xctype(XCType.PYSCF_PARSABLE)  # parsable xc; handle it by normal way
    xc_remains = xc.remove(xc_parsable, inplace=False)  # xc handled in this function
    hyb = ni_original.hybrid_coeff(xc_parsable.token)  # hybrid coefficient from parsable xc
    rsh_coeff = ni_original.rsh_coeff(xc_parsable.token)  # range separate coefficients from parsable xc

    # parse type of current xc
    def get_xc_type(xc_list):
        if len(xc_list.extract_by_xctype(XCType.MGGA)) > 0:
            xc_type = "MGGA"
        elif len(xc_list.extract_by_xctype(XCType.GGA)) > 0:
            xc_type = "GGA"
        elif len(xc_list.extract_by_xctype(XCType.LDA)) > 0:
            xc_type = "LDA"
        else:
            xc_type = "HF"
        return xc_type

    xc_type_full = get_xc_type(xc)  # xctype of full xc code
    xc_type_parsable = get_xc_type(xc_parsable)  # xctype of parsable xc code

    # evaluate parsable xc by normal way
    def eval_xc_eff_parsable(_numint, _xc_code, rho, xctype=xc_type_parsable, *args, **kwargs):
        return ni_original.eval_xc_eff(xc_parsable.token, rho, *args, **kwargs)

    # multiply factor (XCInfo.fac) on generated exc, vxc, fxc, kxc
    def multiply_factor_on_eval_xc_eff(generator, factor):
        def wrapped(*args, **kwargs):
            results = generator(*args, **kwargs)
            for res in results:
                if res is not None:
                    res *= factor
            return results
        return wrapped

    # lists of xc_eff generators
    gen_lists = [eval_xc_eff_parsable]

    # append xc_eff generators
    for xc_info in xc_remains:
        if XCType.SSR in xc_info.type:
            if XCType.EXCH in xc_info.type:
                x_code, omega = xc_info.parameters
                x_fr = xc_info.additional.get("ssr_x_fr", flags["ssr_x_fr"])
                x_sr = xc_info.additional.get("ssr_x_sr", flags["ssr_x_sr"])
                generator = util.eval_xc_eff_ssr_generator(x_code, x_fr, x_sr, omega=omega)
                generator = multiply_factor_on_eval_xc_eff(generator, xc_info.fac)
                gen_lists.append(generator)
            elif XCType.CORR in xc_info.type:
                c_code, omega = xc_info.parameters
                c_fr = xc_info.additional.get("ssr_c_fr", flags["ssr_c_fr"])
                c_sr = xc_info.additional.get("ssr_c_sr", flags["ssr_c_sr"])
                generator = util.eval_xc_eff_ssr_generator(c_code, c_fr, c_sr, omega=omega)
                generator = multiply_factor_on_eval_xc_eff(generator, xc_info.fac)
                gen_lists.append(generator)
            else:
                assert False
        else:
            raise ValueError("Some type of xc is not available!")

    def array_add_with_diff_rows(mat1, mat2):
        assert len(mat1.shape) == len(mat2.shape)
        assert mat1.shape[1:] == mat2.shape[1:]
        if len(mat1) >= len(mat2):
            mat1[:len(mat2)] += mat2
            return mat1
        else:
            mat2[:len(mat1)] += mat1
            return mat2

    def eval_xc_eff(*args, **kwargs):
        exc, vxc, fxc, kxc = gen_lists[0](*args, **kwargs)
        for gen in gen_lists[1:]:
            exc1, vxc1, fxc1, kxc1 = gen(*args, **kwargs)
            if exc is not None:
                exc = array_add_with_diff_rows(exc, exc1)
            if vxc is not None:
                vxc = array_add_with_diff_rows(vxc, vxc1)
            if fxc is not None:
                fxc = array_add_with_diff_rows(fxc, fxc1)
            if kxc is not None:
                kxc = array_add_with_diff_rows(kxc, kxc1)
        return exc, vxc, fxc, kxc

    ni_custom.eval_xc_eff = MethodType(eval_xc_eff, ni_custom)
    ni_custom.hybrid_coeff = lambda *args, **kwargs: hyb
    ni_custom.rsh_coeff = lambda *args, **kwargs: rsh_coeff
    ni_custom._xc_type = lambda *args, **kwargs: xc_type_full
    return ni_custom
