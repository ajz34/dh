import tempfile
import warnings

import h5py
import pickle
import numpy as np
import shutil
import os
from dataclasses import dataclass
from contextlib import contextmanager

from typing import List, Tuple


class RestrictedDataset(h5py.Dataset):
    """
    Writing-Restricted h5py Dataset

    This inheritance aquire that h5py data is only writable when
    ``self.attrs["writeable"]`` is set to True.

    Notes
    -----
    This implementation requires direct modification to ``h5py._hl.dataset``.
    If other programs uses ``h5py.File.create_dataset``, they may actually
    uses this ``RestrictedDataset`` class.
    To avoid possible inconvenience, the attribute ``writeable`` is True
    by default.

    This writeable-protection class is currently not activated.
    To activate this class, please replace original ``h5py.Dataset`` by
    ``h5py._hl.dataset.Dataset = RestrictedDataset``.
    """
    def __init__(self, *args, **kwargs):
        super(RestrictedDataset, self).__init__(*args, *kwargs)
        self.attrs["writeable"] = True

    def __setitem__(self, *args, **kwargs):
        if "writeable" in self.attrs and not self.attrs["writeable"]:
            raise ValueError("assignment destination is read-only")
        super(RestrictedDataset, self).__setitem__(*args, *kwargs)


class HybridDict(dict):
    """
    HybridDict: Inherited dictionary class

    A dictionary specialized to store data both in memory and in disk.

    Parameters
    ----------
    chkfile_name : str
        File name for HDF5 data
    pathdir : str
        File directory for HDF5 data

    Notes
    -----
    This class is inherited from ``dict``, and you surely could store objects other than numpy or h5py tensors.

    But for those objects, you may not utilize the following attribute functions.

    It is recommanded to store tensors directly by key, instead of tuples or lists. For example, to store MO
    coefficients for UHF, use keys ``"mo_coeff(a)": C_a`` and ``"mo_coeff(b)": C_b`` or something similar,
    instead of using ``"mo_coeff": (C_a, C_b)``. Actually, ``"mo_coeff": np.array([C_a, C_b])`` should also be
    acceptable.
    """
    def __init__(self, chkfile_name=None, pathdir=None, **kwargs):
        super(HybridDict, self).__init__(**kwargs)
        # initialize input variables
        if pathdir is None:
            try:
                from pyscf import lib
                pathdir = lib.param.TMPDIR
            except ImportError:
                pathdir = "/tmp/"
        if chkfile_name is None:
            self._chkfile = tempfile.NamedTemporaryFile(dir=pathdir)
            chkfile_name = self._chkfile.name
        # create or open exist chkfile
        self.chkfile_name = chkfile_name
        self.chkfile = h5py.File(self.chkfile_name, "r+")

    def create(self, name, data=None, incore=True, shape=None, dtype=np.float64, **kwargs):
        """
        Create an tensor by h5py style

        There could be two schemes:

        1. Tensor data is already available
        2. Tensor data is note available, but its shape is known

        Note that API user should either provide data or shape info. Providing both of them or none is invalid.

        Parameters
        ----------
        name : str
            Tensor name as key of dictionary
        data : np.array
            Tensor data if scheme 1; otherwise leave it to None
        incore : bool
          True: store tensor in numpy; False: store tensor by h5py to disk
        shape : Tuple[int, ...]
           Tensor shape if scheme 2, tuple like; otherwise leave it to None
        dtype : type
            Tesor data type (np.float32, int, etc)

        Returns
        -------
        np.ndarray or h5py.Dataset
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
        if not incore:
            self.chkfile.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)
            self.setdefault(name, self.chkfile[name])
        elif data is not None:
            self.setdefault(name, np.asarray(data, dtype=dtype))
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
        key : str
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
        Load an array item to memory space (numpy array).

        This is not equilvant to ``dict.get(key)``. For non-array objects, this function
        is meaningless.

        Main purpose of this function is to give a unified function interface to obtain
        numpy array from either numpy, hdf5 or other types of arrays.

        Parameters
        ----------
        key : str
            Key of this item

        Returns
        -------
        np.ndarray
        """
        return np.asarray(self.get(key))

    def apply_func(self, func):
        """
        Apply functin to all values in dictionary.

        As an example, to give types of every item entry of current HybridDict object::

            tensors.apply_func(type)

        Parameters
        ----------
        func : callable
            Function to be applied.

        Returns
        -------
        dict
        """
        types = {}
        for key, val in self.items():
            types[key] = func(val)
        return types

    @staticmethod
    def get_dataset_keys(f):
        """
        Get h5py dataset keys to the bottom level.

        https://stackoverflow.com/a/65924963/7740992

        Parameters
        ----------
        f : h5py.File
            Instance of h5py file.

        Returns
        -------
        List[str]
        """
        keys = []
        f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys

    def dump(self, h5_path="tensors.h5", dat_path="tensors.dat"):
        """
        Dump dictionary to disk.

        There are will be two files: one containing h5py files, another non-h5py dumped by pickle.

        Parameters
        ----------
        h5_path : str
            h5py data path, involving copy operation
        dat_path : str
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
    def pick(h5_path="tensors.h5", dat_path="tensors.dat", **kwargs):
        """
        Load dictionary from disk.

        Additional kwargs are passed to constructor of ``HybridDict``.

        Parameters
        ----------
        h5_path : str
            h5py data path
        dat_path : str
            non-h5py data path

        Returns
        -------
        HybridDict
        """
        tensors = HybridDict(**kwargs)
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


@dataclass
class Params:
    """ Parameters and data.

    This class hope programmers adopting variable protection.
    See documentation for developer.
    """
    flags: dict
    """ Flags stored by dictionary.
    
    Suggests to be composed by simple types such as
    booleans, integers and strings. Should be serializable by pickle.
    """
    tensors: HybridDict
    """ Intermediate tensors.
    
    Large tensors are supposed to be written into this dictionary.
    Should be serializable.
    """
    results: dict
    """ Computed results stored by dictionary.
    
    Should be serializable. Array size of values stored in dictionary should be generally small.
    """

    def __iter__(self):
        yield from [self.flags, self.tensors, self.results]

    @contextmanager
    def fill_default_flag(self, default_flag):
        """ Temporarily fill with default flags.

        User-defined flags will be preserved while flags defined in ``default_flag``
        but not in current flags will be included.
        Use this function by with expression.

        Parameters
        ----------
        default_flag : dict
            Default flags to be filled.
        """
        old_flags = self.flags.copy()
        self.flags = default_flag
        self.flags.update(old_flags)
        yield self
        self.flags = old_flags

    @contextmanager
    def temporary_flags(self, add_flags):
        """ Temporarily additional flags.

        Use this function by with expression::

            with params.temporary_flags({"key": value}):
                print(params.flags)

        Parameters
        ----------
        add_flags : dict
            Temporarily changed flags
        """
        old_flags = self.flags.copy()
        self.flags.update(add_flags)
        yield self
        self.flags = old_flags

    def update_results(self, income_result, allow_overwrite=True, warn_overwrite=True):
        """ Update results.

        This function may change attribute ``self.results``.

        Parameters
        ----------
        income_result : dict
            Result dictionary to be updated into ``self.results``.
        allow_overwrite : bool
            Whether allows overwriting result dictionary.
        warn_overwrite : bool
            Whether warns overwriting result dictionary.
        """
        if not allow_overwrite or warn_overwrite:
            keys_interset = set(income_result).intersection(self.results)
            if len(keys_interset) != 0:
                if not allow_overwrite:
                    raise KeyError("Overwrite results is not allowed!\n"
                                   "Repeated keys: [{:}]".format(", ".join(keys_interset)))
                if warn_overwrite:
                    msg = "Key result overwrited!\n"
                    for key in keys_interset:
                        msg += "Key: `{:}`, before {:}, after {:}".format(key, income_result[key], self.results[key])
                        warnings.warn(msg)
        self.results.update(income_result)
        return self
