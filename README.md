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
print(mf.e_tot)
```

Name `dh` refers to **D**oubly **H**ybrid.
`DFDH` refers to **D**ensity **F**itting **D**oubly **H**ybrid.

### Geometric optimization

```python
from pyscf import gto, dh
from pyscf.geomopt.berny_solver import optimize  # or geometric_solver
mol = gto.Mole(atom="O; H 1 1.0; H 1 1.0 2 104.5", basis="cc-pVDZ", verbose=0).build()
mf = dh.DFDH(mol, xc="XYG3").nuc_grad_method()
mol_eq = optimize(mf)  # optimized molecule object
```

### Polarizability

```python
from pyscf import gto, dh
mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ", verbose=0).build()
mf = dh.DFDH(mol, xc="XYG3").polar_method().run()
print(mf.pol_tot)
```

Hessian is currently not implemented.

## Install

Refer to installation of PySCF [Extension modules](https://pyscf.org/install.html#extension-modules).
Declare `PYSCF_EXT_PATH=$PYSCF_EXT_PATH:/path/to/dh` should work.

This extension overwhelmingly relies on [pyscf-tblis](https://github.com/pyscf/pyscf-tblis).
Also recommands modifing `EINSUM_MAX_SIZE` (in
[tblis_einsum.py](https://github.com/pyscf/pyscf-tblis/blob/160333ab28d0d9c6900bd5b77efc8d03dd1c74c5/pyscf/tblis_einsum/tblis_einsum.py#L45))
to much smaller value for `dh` extension.

## Availibility and Limitations

- Supported doubly hybrid functionals (and MP2):
    - MP2
    - Early XYG3 family:
      [XYG3](https://doi.org/10.1073/pnas.0901093106),
      [XYGJ-OS](https://doi.org/10.1073/pnas.1115123108),
      [xDH-PBE0](https://dx.doi.org/10.1063/1.3703893)
    - [xDH@B3LYP family](https://dx.doi.org/10.1021/acs.jpclett.1c00360):
      revXYG3, revXYGJ-OS, XYG5, XYGJ-OS5, XYG6, XYG7
    - B2P family:
      [B2PLYP](http://dx.doi.org/10.1063/1.2148954),
      [B2GPPLYP](https://doi.org/10.1021/jp801805p),
      [mPW2PLYP](http://dx.doi.org/10.1039/B608478H)
    - PBE related:
      [PBE0-DH](http://dx.doi.org/10.1063/1.3604569),
      [LS1DH-PBE](https://doi.org/10.1063/1.3640019),
      [PBE0-2](https://doi.org/10.1016/j.cplett.2012.04.045),
      [PBE-QIDH](http://dx.doi.org/10.1063/1.4890314),
    - Self-defined functionals
      (only supports pure HF or hybrid GGA functionals, if gradient/electric property is required)
    
Default functional is XYG3 currently.


### Near Future Plans

- Another independent module (maybe called `dheng`) handling only energy evaluation for
  other type of doubly-hybrid functionals,
  such as D3(BJ) dispersion, Laplace-transformation,
  long-range corrected PT2, renormalization based methods,
  random-phase-approximation (RPA) based methods, etc.;
- Quadrupole;
- Rectify APIs, and more formal logging and timing;
- API document and user document;
- Efficiency benchmarking;
- D3(BJ) dispersion derivative properties.

### Future Plans?

- Hessian and dipole-derivative;
- Frozen core;
- RIJONX, RICOSX and conventional SCF method supports;
- Laplace-transformation method derivatives;
- PBC energy;
- Other derivative properties ...

## Bibliography

- For functional energy evaluation, refers to the origin paper of those functionals.  

- For first order properties (atom nuclear gradient, dipole moment):

  > Neil Qiang Su, Igor Ying Zhang, Xin Xu.
  > *J. Comput. Chem.* **2013**, *34*, 1759-1774.
  > doi: [10.1002/jcc.23312](https://dx.doi.org/10.1002/jcc.23312)
  > 
  > Analytic Derivatives for the XYG3 Type of Doubly
  > Hybrid Density Functionals: Theory, Implementation,
  > and Assessment

- For second order properties
  (hessian of xDH obtained from conventional SCF/PT2 in Gaussian 09,
  where polarizability is also implemented)
  that relates to this work:

  > Yonghao Gu, Zhenyu Zhu, Xin Xu. submitted.

Predecessor of this work is [Py_xDH](https://github.com/ajz34/Py_xDH).
