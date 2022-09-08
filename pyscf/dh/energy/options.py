from enum import Enum


frozen_rule = None
"""
Rule for frozen orbital numbers.

This option will be generate and be overrided by option ``frozen_list``.

Default to None.

Warnings
--------
TODO: Function to parse frozen rule.
"""

frozen_list = None
"""
Index list of frozen orbitals.

For example, if set to ``[0, 2, 3, 4]``, then those orbitals are not correlated in MP2 calculation.

Default to None.
"""


class incore_t_ijab(Enum):
    """
    Flag for tensor :math:`t_{ij}^{ab}` stored in memory or disk.
    """
    true = True
    """Store tensor in memory."""
    false = False
    """Store tensor in disk."""
    auto = "auto"
    """Leave program to judge whether tensor locates."""
    default = auto
    """Default to auto."""
