from pyscf import dh, gto, scf
from pyscf.dh.util import Params, HybridDict, default_options
import numpy as np


def test_rmp2_conv():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
    mf_s = scf.RHF(mol).run()

    mf = dh.energy.RDH(mf_s)
    mf.params = Params(default_options, HybridDict(), {})
    with mf.params.temporary_flags({"integral_scheme": "conv"}):
        mf.driver_energy_mp2()
    print(mf.params.results)
    assert np.allclose(mf.params.results["eng_mp2"], -0.12722216721906485)
