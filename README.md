# Object-Oriented Version of 99 lines Topology Optimization


## General

Object-oriented and `numba` optimized version of "99 lines Topology optimization in Matlab"
Python version. Main objective is simply is to enhance numerical experiments when changing
topology optimization parameters and boundart condition and load cases.

## Example

```python
from TopolSettings import SetTopol
```

This imports all necessary classes and modules to set and launch a new topology optimization

```python
top = SetTopol()
top.optimize()
```
