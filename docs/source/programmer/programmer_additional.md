# Additional Notes For Programmers

This is only scratch. No guarantee that these guidelines could work.

## Python extension with PyCharm inspection

Please make the following statements at top of `pyscf/__init__.py` (change pyscf directly instead of dh):
```python
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
```
Then `import pyscf.dh` would not show error.
This is related to https://youtrack.jetbrains.com/issue/PY-38434.
