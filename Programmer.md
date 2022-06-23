# Programming Rules for program dh (scratch)

This is only scratch. No guarantee that these guidelines could work.

## Dictionary based data scheme
   
All data should be transfered by instance of

```python
param = Param(flags, tensors, results)
```

1. `flags` contains running configurations. Usual dictionary.
   - Should be read-only for most situations, but no enforcement on this.
     If API user need to pass additional flags in functions,
     use `with param.with_flags_add({"add_flag": add_val})`.
   - Should contain simple types such as booleans, integers, enums, or tuples of those types.
     `flags` is at least serializable.
2. `tensors` contains intermediate matrices/tensors. `TensorDict` instance.
   - Should contain `h5py` instance, `np.ndarray` instance, or `Tuple(np.ndarray)` or `List(np.ndarray)`.
     Other types are strongly not recommended.
   - Scalar values are recommanded to be stored in `results` instead of `tensors`; user could also transfer scalar
     value to `np.ndarray` instance.
   - Should be read-only for most times. If API user need to change values,
     use `with param.with_tensors_write(["tensor_to_be_written_1", "tensor_2"])`.
   - Terminal users of this program should have no access to write any tensor.
3. `results` contains outputs.
   - `results` should be serializable.
   
## Functional programming

For most computing extensive processes, functional programming should be adopted.

These kind of functions should have the following signature:

```python
def func(param, opt1, opt2, ...):
    """
    Output Tensors
    --------------
    tensor1
        Use of tensor1
    """
```

It is possible to implicitly pass input variables from `param.tensors`. However, passing tensors
by arguments is more prefered.

```python
# instead of using
def func(param):
    return param["foo"].sum()
# the following is more preferred:
def func_more_preferred(_param, foo):
    # `_` before variable in signature tells PyCharm that variable could be unused
    return foo.sum()
```

Then OOP (object-oriented programs) wraps computing extensive functions. For example (though probably not suitable):

```python
def proc_dm(self, mo_coeff=None):  # defined as class member function
    if mo_coeff is None: mo_coeff = self.param.tensors["mo_coeff"]
    dm = comp_dm(self.param, mo_coeff)
    self.param.tensors.create("dm", data=dm)
    
def comp_dm(param, mo_coeff):  # computing extensive function
    nocc = param.flags["nocc"]
    return 2 * mo_coeff[:, :nocc] @ mo_coeff[:, :nocc].conj().T
```

## Python extension with PyCharm inspection

Please make the following statements at top of `pyscf/__init__.py` (change pyscf directly instead of dh):
```python
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
```
Then `import pyscf.dh` would not show error.
This is related to https://youtrack.jetbrains.com/issue/PY-38434.
