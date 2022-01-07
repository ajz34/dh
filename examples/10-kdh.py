#!/usr/bin/env python

'''
KDH at an individual k-point
'''

from functools import reduce
import numpy
from pyscf.pbc import gto
from pyscf import pbcdh, lib

lib.num_threads(28)

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

#
# DF-KDH with 2x2x2 k-points
#
kpts = cell.make_kpts([2,2,2])
#kpts = cell.make_kpts([4,4,4])
#kmf = scf.KRHF(cell)#.rs_density_fit()
#kmf.kpts = kpts
#ehf = kmf.kernel()

mypt = pbcdh.KDH(cell, xc="XYG3", kpts=kpts)
mypt.max_memory = 10000
mypt.kernel()
print("PBC-XYG3 energy (per unit cell) =", mypt.e_tot)

