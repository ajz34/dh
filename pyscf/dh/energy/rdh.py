from pyscf import lib, dft
from pyscf.dh.util import Params, HybridDict, default_options
from pyscf.dh.energy.rmp2 import driver_energy_mp2


class RDH(lib.StreamObject):
    """ Restricted doubly hybrid class.
    """
    mf_s: dft.rks.RKS
    """ Self consistent object. """
    mf_n: dft.rks.RKS
    """ Energy computation object. """
    params: Params
    """ Params object consisting of flags, tensors and results. """
    def __init__(self, mf_s=NotImplemented, mf_n=NotImplemented, params=None):
        self.mf_s = mf_s
        self.mf_n = mf_n
        if params:
            self.params = params
        else:
            self.params = Params(default_options, HybridDict(), {})

    driver_energy_mp2 = driver_energy_mp2
