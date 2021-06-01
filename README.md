# dh

Doubly hybrid methods for PySCF.

SCF must run in RI-JK currently. Correlation part (PT2 part) is definitely density fitting.
Runs in disk-based way by default.

## Usage

### Energy evaluation

```python
from pyscf import gto, dh
mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", verbose=0).build()
mf = dh.DFDH(mol, xc="XYG3").run()
print(mf.e_tot)  # -76.36230265411723 in Hartree
```

Name `dh` refers to **D**oubly **H**ybrid.
`DFDH` refers to **D**ensity **F**itting **D**oubly **H**ybrid.

### Geometric optimization

```python
from pyscf import gto, dh, data
from pyscf.geomopt.berny_solver import optimize  # or geometric_solver
mol = gto.Mole(atom="O; H 1 1.0; H 1 1.0 2 104.5", basis="cc-pVDZ", verbose=0).build()
mf = dh.DFDH(mol, xc="XYG3").nuc_grad_method()
mol_eq = optimize(mf)  # optimized molecule object
# print geom coordinates in Angstrom
print(mol_eq.atom_coords() * data.nist.BOHR)
# [[ 0.00573557  0.          0.00740783]
#  [ 0.9649555   0.          0.02121004]
#  [-0.22107107  0.          0.93952977]]
```

### Polarizability

```python
from pyscf import gto, dh
mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", verbose=0).build()
mf = dh.DFDH(mol, xc="XYG3").polar_method().run()
print(mf.pol_tot.trace() / 3)  # 4.9747082620200915 in a.u.
```

Hessian is currently not implemented.

## Install

### `dh` as PySCF extension
Refer to installation of PySCF [Extension modules](https://pyscf.org/install.html#extension-modules).
Declare `PYSCF_EXT_PATH=$PYSCF_EXT_PATH:/path/to/dh` should work.

### `pyscf-tblis` extension
This extension overwhelmingly relies on [pyscf/pyscf-tblis](https://github.com/pyscf/pyscf-tblis).
Also recommands modifing `EINSUM_MAX_SIZE` (in
[tblis_einsum.py](https://github.com/pyscf/pyscf-tblis/blob/160333ab28d0d9c6900bd5b77efc8d03dd1c74c5/pyscf/tblis_einsum/tblis_einsum.py#L45))
and `lib_einsum_max_size` (in your [PySCF config](https://pyscf.org/install.html#cmake-options-and-compiling-flags), refer to `PYSCF_CONFIG_FILE`)
to much smaller value for `dh` extension.

### `dftd3` extension
If you are not using functionals with "-D3" suffix, you can safely omit this part.

To calculate functionals that includes D3 dispersion correction, user may need to install
[pyscf/dftd3]() as an extension of PySCF. 

Furthermore, this extension requires a dynamic library `libdftd3.so`, which should be compiled by user.
- Source code of the library could be accessed from [ajz34/libdftd3](https://github.com/ajz34/libdftd3);
- Make the library in folder `lib`;
- Copy the generated file `libdftd3.so` to `path_to_pyscf_dftd3/pyscf/dftd3`, where `itrf.py` exists.

A reminder to experienced PySCF users is that, the library
[ajz34/libdftd3](https://github.com/ajz34/libdftd3) is not identical to it's original form
[cuanto/libdftd3](https://github.com/cuanto/libdftd3). So please do not copy your old `libdftd3.so`
to extension `pyscf/dftd3`.


## Availibility and Limitations

- Supported features:
  Res/Unrestricted single point energy, gradient, static polarizability
  for XYG3-like and B2PLYP-like doubly functionals
- Supported doubly hybrid functional (and MP2, certainly) keywords:
    - MP2
    - Early XYG3 family:
      [XYG3](https://doi.org/10.1073/pnas.0901093106),
      [XYGJ-OS](https://doi.org/10.1073/pnas.1115123108),
      [xDH-PBE0](https://dx.doi.org/10.1063/1.3703893)
    - [xDH@B3LYP family](https://dx.doi.org/10.1021/acs.jpclett.1c00360):
      revXYG3, revXYGJ-OS, XYG5, XYGJ-OS5, XYG6, XYG7
    - B2P family:
      [B2PLYP](http://dx.doi.org/10.1063/1.2148954),
      [B2PLYP-D3](https://doi.org/10.1063/1.3382344),
      [B2GPPLYP](https://doi.org/10.1021/jp801805p),
      [mPW2PLYP](http://dx.doi.org/10.1039/B608478H)
    - PBE related:
      [PBE0-DH](http://dx.doi.org/10.1063/1.3604569),
      [LS1DH-PBE](https://doi.org/10.1063/1.3640019),
      [PBE0-2](https://doi.org/10.1016/j.cplett.2012.04.045),
      [PBE-QIDH](http://dx.doi.org/10.1063/1.4890314)
    - [DSD family](https://doi.org/10.1002/jcc.23391)
      with D3(BJ) dispersion correction (version 2013):
      DSD-PBEP86-D3, DSD-PBEPBE-D3, DSD-BLYP-D3, DSD-PBEB95-D3
    - Non-consistent functionals:
      HF-B3LYP, HF-PBE0 (no PT2 contribution)
    - Self-defined functionals
      (only supports pure HF or hybrid GGA functionals, if gradient/electric property is required)
      
      Format: `("xc_SCF", "xc_energy", coef_PT2, coef_OS_PT2, coef_SS_PT2)`
    
Default functional is XYG3 currently.

**Warning**: Energy of DFT-D3(BJ) dispersion functionals haven't been tested and compared
to other softwares thoroughly!

### Near Future Plans

- Quadrupole;
- Rectify APIs, and more formal logging and timing;
- API document and user document;
- Efficiency benchmarking
    - Preliminary tests shows good performance for small molecules with large basis sets;
    - Polarizability should gain much speedup due to its algorithm implementation.

### Future Plans?

- Another independent module (maybe called `dheng`) handling only energy evaluation for
  more doubly-hybrid functionals,
  such as Laplace-transformation,
  long-range corrected PT2, renormalization based methods,
  random-phase-approximation (RPA) based methods, etc.;
- Hessian and dipole-derivative;
- Frozen core;
- RIJONX, RICOSX and conventional SCF method supports;
- Laplace-transformation method derivatives;
- PBC energy;
- Other derivative properties ...

## Bibliography

- For functional energy evaluation, refers to the origin paper of those functionals.  

- For first order properties (xDH atom nuclear gradient, dipole moment first implemented
  in local NWChem):

  > Neil Qiang Su, Igor Ying Zhang, Xin Xu.
  > *J. Comput. Chem.* **2013**, *34*, 1759-1774.
  > doi: [10.1002/jcc.23312](https://dx.doi.org/10.1002/jcc.23312)
  > 
  > Analytic Derivatives for the XYG3 Type of Doubly
  > Hybrid Density Functionals: Theory, Implementation,
  > and Assessment

- For second order properties
  (hessian of xDH obtained from conventional SCF/PT2 in local Gaussian 09,
  where polarizability is also implemented)
  that relates to this work:

  > Yonghao Gu, Zhenyu Zhu, Xin Xu. submitted.

Predecessor of this work is [Py_xDH](https://github.com/ajz34/Py_xDH).
