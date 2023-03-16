from pyscf import dft


def eval_xc_eff_ssr_generator(name_code, name_fr, name_sr, omega=0.7, cutoff=1e-13):
    """ Generator for function of scaled short-range DFT integrals.

    Parameters
    ----------
    name_code : str
        xc functional to be scaled.
    name_fr : str
        Full-range functional of the scale. Must be LDA.
    name_sr : str
        Short-range functional of the scale. Must be LDA.
    omega : float
        Range-separate parameter.
    cutoff : float
        Cutoff of exc_fr. If grid value of exc_fr is too small, then the returned exc on this grid will set to be zero.

    Returns
    -------
    callable
        A function shares the same function signature with dft.numint.NumInt.eval_xc_eff.
    """

    # original numint object
    ni = dft.numint.NumInt()
    # currently only LDA is supported
    assert ni._xc_type(name_fr) == "LDA"
    assert ni._xc_type(name_sr) == "LDA"
    # the xc type concern in this scaled short-range term
    xc_type_code = ni._xc_type(name_code)

    def eval_xc_eff(numint, xc_code, rho, xctype=None, *args, **kwargs):
        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == 'LDA':
            spin_polarized = rho.ndim >= 2
        else:
            spin_polarized = rho.ndim == 3

        if not spin_polarized:
            return eval_xc_eff_spin_nonpolarized(numint, xc_code, rho, xctype=xctype, *args, **kwargs)
        else:
            return eval_xc_eff_spin_polarized(numint, xc_code, rho, xctype=xctype, *args, **kwargs)

    def eval_xc_eff_spin_nonpolarized(numint, xc_code, rho, deriv=1, omega=omega, xctype=None, *_args, **_kwargs):
        # the xc type of the whole numint object
        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == "LDA":
            rho0 = rho.copy()
        else:
            rho0 = rho[0].copy()
        # if the xc type concern is LDA, then only extract the density grid (instead of its derivatives)
        if xc_type_code == "LDA":
            rho = rho0
        # evaluate xc grids by original numint object
        exc_code, vxc_code, fxc_code, kxc_code = ni.eval_xc_eff(name_code, rho, deriv=deriv)
        exc_fr, vxc_fr, fxc_fr, kxc_fr = ni.eval_xc_eff(name_fr, rho0, deriv=deriv)
        exc_sr, vxc_sr, fxc_sr, kxc_sr = ni.eval_xc_eff(name_sr, rho0, deriv=deriv, omega=omega)
        # avoid too small denominator (must set grid values to zero on these masks)
        mask = abs(exc_fr) < cutoff
        exc_fr[mask] = cutoff
        rho0[mask] = cutoff
        # handle exc, vxc, fxc, kxc
        ratio = exc_sr / exc_fr
        exc = vxc = fxc = kxc = None
        if deriv >= 0:
            exc = exc_code * ratio
            exc[mask] = 0
        if deriv >= 1:
            vxc = vxc_code.copy()
            vxc[0] = (
                + vxc_code[0] * exc_sr / exc_fr
                + exc_code * vxc_sr[0] / exc_fr
                - exc_code * exc_sr * vxc_fr[0] / exc_fr ** 2
            )
            vxc[1:] *= ratio
            vxc[:, mask] = 0
        if deriv >= 2:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            fxc = fxc_code.copy()
            r = rho0
            c, dc, ddc = exc_code * r, vxc_code[0], fxc_code[0, 0]
            s, ds, dds = exc_sr * r, vxc_sr[0], fxc_sr[0, 0]
            f, df, ddf = exc_fr * r, vxc_fr[0], fxc_fr[0, 0]
            fxc *= ratio
            fxc[0, 0] = (
                + (c * dds + 2 * dc * ds + ddc * s) / f
                - (c * s * ddf + 2 * c * ds * df + 2 * dc * s * df) / f**2
                + 2 * c * s * df**2 / f**3
            )
            fxc[:, :, mask] = 0
        if deriv >= 3:
            raise NotImplemented("fxc and kxc for scaled short-range functionals are not implemented.")
        return exc, vxc, fxc, kxc

    def eval_xc_eff_spin_polarized(numint, xc_code, rho, deriv=1, omega=omega, xctype=None, *_args, **_kwargs):
        raise NotImplementedError("Spin-polarized is still not implemented!")

    return eval_xc_eff
