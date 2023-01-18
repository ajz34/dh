from pyscf import lib
import numpy as np
from typing import List

from pyscf.dh.energy.rdh import RDH
from pyscf.dh.energy.ump2 import driver_energy_ump2
from pyscf.dh.energy.uiepa import driver_energy_uiepa
from pyscf.dh import util
from pyscf.dh.energy.udft import kernel_energy_unrestricted_exactx, kernel_energy_unrestricted_noxc


class UDH(RDH):
    """ Unrestricted doubly hybrid class. """

    @property
    def restricted(self) -> bool:
        return False

    @property
    def nocc(self) -> List[int]:
        """ Number of occupied (alpha, beta) molecular orbitals. """
        return list(self.mol.nelec)

    @property
    def nvir(self) -> List[int]:
        """ Number of virtual (alpha, beta) molecular orbitals. """
        return [self.nmo - self.nocc[0], self.nmo - self.nocc[1]]

    def get_mask_act(self, regenerate=False) -> np.ndarray:
        """ Get mask of active orbitals.

        Dimension: (2, nmo), boolean array
        """
        if regenerate or "mask_act" not in self.params.tensors:
            frozen_rule = self.params.flags["frozen_rule"]
            frozen_list = self.params.flags["frozen_list"]
            mask_act = util.parse_frozen_list(self.mol, self.nmo, frozen_list, frozen_rule)
            if len(mask_act.shape) == 1:
                # enforce mask to be [mask_act_alpha, mask_act_beta]
                mask_act = np.array([mask_act, mask_act])
            self.params.tensors["mask_act"] = mask_act
        return self.params.tensors["mask_act"]

    @property
    def nmo_f(self) -> List[int]:
        return [self.get_mask_act()[s].sum() for s in (0, 1)]

    @property
    def nocc_f(self) -> List[int]:
        return [self.get_mask_act()[s][:self.nocc[s]].sum() for s in (0, 1)]

    @property
    def nvir_f(self) -> List[int]:
        return [self.get_mask_act()[s][self.nocc[s]:].sum() for s in (0, 1)]

    @property
    def mo_coeff_f(self) -> List[np.ndarray]:
        return [self.mo_coeff[s][:, self.get_mask_act()[s]] for s in (0, 1)]

    @property
    def mo_energy_f(self) -> List[np.ndarray]:
        return [self.mo_energy[s][self.get_mask_act()[s]] for s in (0, 1)]

    def get_Y_ov_f(self, regenerate=False) -> List[np.ndarray]:
        """ Get cholesky decomposed ERI in MO basis (occ-vir part with frozen core).

        Dimension: (naux, nocc_f, nvir_f) for each spin.
        """
        nmo_f, nocc_f = self.nmo_f, self.nocc_f
        if regenerate or "Y_ov_f" not in self.params.tensors:
            Y_ov_f = [util.get_cderi_mo(
                self.df_ri, self.mo_coeff_f[s], None, (0, nocc_f[s], nocc_f[s], nmo_f[s]),
                self.mol.max_memory - lib.current_memory()[0]
            ) for s in (0, 1)]
            self.params.tensors["Y_ov_f_a"] = Y_ov_f[0]
            self.params.tensors["Y_ov_f_b"] = Y_ov_f[1]
        else:
            Y_ov_f = [self.params.tensors["Y_ov_f_a"], self.params.tensors["Y_ov_f_b"]]
        return Y_ov_f

    driver_energy_mp2 = driver_energy_ump2
    driver_energy_iepa = driver_energy_uiepa

    kernel_energy_exactx = staticmethod(kernel_energy_unrestricted_exactx)
    kernel_energy_noxc = staticmethod(kernel_energy_unrestricted_noxc)
