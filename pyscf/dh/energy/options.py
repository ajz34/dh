frozen_rule = None
""" Rule for frozen orbital numbers.

This option will be generate and be overrided by option ``frozen_list``.

Default to None.

Warnings
--------
TODO: Function to parse frozen rule.
"""

frozen_list = None
""" Index list of frozen orbitals.

For example, if set to ``[0, 2, 3, 4]``, then those orbitals are not correlated in MP2 calculation.

Default to None.
"""

incore_t_ijab = "auto"
""" Flag for tensor :math:`t_{ij}^{ab}` stored in memory or disk.

Parameters
----------
True
    Store tensor in memory.
False
    Store tensor in disk.
"auto"
    Leave program to judge whether tensor locates.
(int)
    If tensor size exceeds this size (in MBytes), then store in disk. 
"""

coef_mp2_os = 1
""" Coefficient of opposite-spin contribution to MP2 energy. """

coef_mp2_ss = 1
""" Coefficient of same-spin contribution to MP2 energy. """
