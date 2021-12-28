#!/usr/bin/env python

'''
KDH at an individual k-point
'''

from functools import reduce
import numpy
from pyscf.pbc import gto
from pyscf import pbcdh

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
# KHF and KMP2 with 2x2x2 k-points
#
kpts = cell.make_kpts([2,2,2])
#kmf = scf.KRHF(cell)#.rs_density_fit()
#kmf.kpts = kpts
#ehf = kmf.kernel()

mypt = pbcdh.KDH(cell, xc="XYG3", kpts=kpts)
mypt.max_memory = 10000
mypt.kernel()
print("KXYG3 energy (per unit cell) =", mypt.e_tot)

exit()
#
# The KHF and KMP2 for single k-point calculation.
#
kpts = cell.get_abs_kpts([0.25, 0.25, 0.25])
kmf = scf.KRHF(cell)
kmf.kpts = kpts
ehf = kmf.kernel()

mypt = mp.KMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)

