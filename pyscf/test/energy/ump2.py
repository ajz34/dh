from pyscf import dh, gto, scf, df, lib, mp
import numpy as np


def test_rmp2_conv():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()
    mf_s = scf.UHF(mol).run()

    mf = dh.energy.UDH(mf_s)
    with mf.params.temporary_flags({"integral_scheme": "conv"}):
        mf.driver_energy_mp2()
    print(mf.params.results)
    assert np.allclose(mf.params.results["eng_mp2"], -0.209174918573074)


def test_rmp2_conv_fc():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()
    mf_s = scf.UHF(mol).run()

    mf = dh.energy.UDH(mf_s)
    mf.params.flags["frozen_rule"] = "FreezeNobleGasCore"
    with mf.params.temporary_flags({"integral_scheme": "conv"}):
        mf.driver_energy_mp2()
    print(mf.params.results)
    assert np.allclose(mf.params.results["eng_mp2"], -0.195783018787701)
