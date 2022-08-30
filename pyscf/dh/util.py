from pyscf import lib, df
from pyscf.ao2mo import _ao2mo

import tempfile
import h5py
import pickle
import numpy as np
import shutil
import os
import dataclasses


class HybridDict(dict):
    """
    HybridDict: Inherited dictionary class

    A dictionary specialized to store data both in memory and in disk.

    Parameters
    ----------
    chkfile_name : str
        File name
    pathdir
        File directory

    Notes
    -----
    This class is inherited from ``dict``, and you surely could store objects other than numpy or h5py tensors.

    But for those objects, you may not utilize the following attribute functions.
    """
    def __init__(self, chkfile_name=None, pathdir=None, **kwargs):
        super(HybridDict, self).__init__(**kwargs)
        # initialize input variables
        if pathdir is None:
            pathdir = lib.param.TMPDIR
        if chkfile_name is None:
            self._chkfile = tempfile.NamedTemporaryFile(dir=pathdir)
            chkfile_name = self._chkfile.name
        # create or open exist chkfile
        self.chkfile_name = chkfile_name
        self.chkfile = h5py.File(self.chkfile_name, "r+")

    def create(self, name, data=None, incore=True, shape=None, dtype=None, **kwargs):
        """
        Create an tensor by h5py style

        There could be two schemes:

        1. Tensor data is already available
        2. Tensor data is note available, but its shape is known

        Note that API user should either provide data or shape info. Providing both of them or none is invalid.

        Parameters
        ----------
        name
            Tensor name as key of dictionary
        data
            Tensor data if scheme 1; otherwise leave it to None
        incore
          True: store tensor in numpy; False: store tensor by h5py to disk
        shape
           Tensor shape if scheme 2, tuple like; otherwise leave it to None
        dtype
            Tesor data type (np.float32, int, etc)

        Returns
        -------
        The tensor you've created.
        """
        # create logic check
        if data is None and shape is None:
            raise ValueError("Provide either data or shape!")
        if data is not None and shape is not None:
            raise ValueError("Data and shape shouldn't be provided together!")
        if name in self:
            try:  # don't create a new space if tensor already exists
                # data provided or shape not aligned is not considered here
                if shape and isinstance(self[name], h5py.Dataset) == (not incore):
                    if self[name].shape == shape:
                        self[name][:] = 0
                        return self.get(name)
            except (ValueError, AttributeError):
                # ValueError -- in h5py.h5d.create: Unable to create dataset (name already exists)
                # AttributeError -- [certain other type] object has no attribute 'shape'
                pass
            self.delete(name)
        dtype = dtype if dtype is not None else np.float64
        if not incore:
            self.chkfile.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)
            self.setdefault(name, self.chkfile[name])
        elif data is not None:
            self.setdefault(name, data)
        elif data is None and shape is not None:
            self.setdefault(name, np.zeros(shape=shape, dtype=dtype))
        else:
            raise ValueError("Could not handle create!")
        return self.get(name)

    def delete(self, key):
        """
        Delete an item.

        Parameters
        ----------
        key
            Key of this item

        Notes
        -----
        If the item to be deleted is an h5py dataset, then the data on disk will also be deleted.
        However, h5py may not handle deleting an item delicately, meaning that deleted item does
        not necessarily (usually not) free the space you may expected.

        Also note that an item could be deleted multiple times without warning.
        """
        val = self.pop(key)
        if isinstance(val, h5py.Dataset):
            try:
                del self.chkfile[key]
            except KeyError:  # h5py.h5g.GroupID.unlink: Couldn't delete link
                # another key maps to the same h5py dataset value, and this value has been deleted
                pass

    def load(self, key):
        """
        Load an item **to memory space**.

        This is not equilvant to ``dict.get(key)``.

        Parameters
        ----------
        key
            Key of this item

        Returns
        -------
        A numpy object
        """
        return np.asarray(self.get(key))

    def dump(self, h5_path="tensors.h5", dat_path="tensors.dat"):
        """
        Dump dictionary to disk.

        There are will be two files: one containing h5py files, another non-h5py dumped by pickle.

        Parameters
        ----------
        h5_path
            h5py data path, involving copy operation
        dat_path
            non-h5py data path, dumped by pickle
        """
        dct = {}
        for key, val in self.items():
            if not isinstance(val, h5py.Dataset):
                dct[key] = val
        with open(dat_path, "wb") as f:
            pickle.dump(dct, f)
        self.chkfile.close()
        shutil.copy(self.chkfile_name, h5_path)
        self.chkfile = h5py.File(self.chkfile_name, "r+")
        # re-update keys stored on disk
        for key in HybridDict.get_dataset_keys(self.chkfile):
            self[key] = self.chkfile[key]

    @staticmethod
    def get_dataset_keys(f):
        # get h5py dataset keys to the bottom level https://stackoverflow.com/a/65924963/7740992
        keys = []
        f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys

    @staticmethod
    def pick(h5_path, dat_path):
        """
        Load dictionary from disk.

        Parameters
        ----------
        h5_path
            h5py data path
        dat_path
            non-h5py data path

        Returns
        -------
        Retrived dictionary.
        """
        tensors = HybridDict()
        tensors.chkfile.close()
        file_name = tensors.chkfile_name
        os.remove(file_name)
        shutil.copyfile(h5_path, file_name)
        tensors.chkfile = h5py.File(file_name, "r+")

        for key in HybridDict.get_dataset_keys(tensors.chkfile):
            tensors[key] = tensors.chkfile[key]

        with open(dat_path, "rb") as f:
            dct = pickle.load(f)
        tensors.update(dct)
        return tensors


@dataclasses.dataclass
class Params:
    flags: dict
    tensors: HybridDict
    results: dict

    def __iter__(self):
        yield from [self.flags, self.tensors, self.results]

    @property
    def temporary_add_flag(self):
        orig_params = self

        class ParamsTmp:
            def __init__(self, add_flags):
                self.orig_flags = orig_params.flags.copy()
                self.add_flags = add_flags

            def __enter__(self):
                orig_params.flags.update(self.add_flags)
                return orig_params

            def __exit__(self, exc_type, exc_val, exc_tb):
                orig_params.flags = self.orig_flags

        return ParamsTmp


def calc_batch_size(unit_flop, mem_avail, pre_flop=0):
    # mem_avail: in MB
    if unit_flop == 0: return 1
    max_memory = 0.8 * mem_avail - pre_flop * 8 / 1024 ** 2
    batch_size = int(max(max_memory // (unit_flop * 8 / 1024 ** 2), 1))
    return batch_size


def get_cderi_mo(with_df: df.DF, C, Y_mo=None, pqslice=None, max_memory=2000):
    naux = with_df.get_naoaux()
    nmo = C.shape[-1]
    if pqslice is None:
        pqslice = (0, nmo, 0, nmo)
        nump, numq = nmo, nmo
    else:
        nump, numq = pqslice[1] - pqslice[0], pqslice[3] - pqslice[2]
    if Y_mo is None:
        Y_mo = np.empty((naux, nump, numq))

    p0, p1 = 0, 0
    preflop = 0 if not isinstance(Y_mo, np.ndarray) else Y_mo.size
    nbatch = calc_batch_size(2*nump*numq, max_memory, preflop)
    for Y_ao in with_df.loop(nbatch):
        p1 = p0 + Y_ao.shape[0]
        Y_mo[p0:p1] = _ao2mo.nr_e2(Y_ao, C, pqslice, aosym="s2", mosym="s1").reshape(p1-p0, nump, numq)
        p0 = p1
    return Y_mo


def gen_leggauss_0_inf(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (1 + x) / (1 - x), w / (1 - x)**2


def gen_leggauss_0_1(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (x + 1), 0.5 * w


if __name__ == '__main__':
    params = Params({}, HybridDict(), {})
    with params.temporary_add_flag({"Add": 1}) as p:
        print(p.flags)
        print(params.flags)
    print(params.flags)
