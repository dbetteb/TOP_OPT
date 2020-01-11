# Object-Oriented Version of 99 lines Topology Optimization


## General

Object-oriented and `numba` optimized version of "99 lines Topology optimization in Matlab"
Python version. Main objective is simply is to enhance numerical experiments when changing
topology optimization parameters and boundart condition and load cases.

## Example

```python
from SetTopol import TopolSettings  
```

This imports all necessary classes and modules to set and launch a new topology optimization

```python
top = TopolSettings()
top.optimize()
```
This launch a topology optimization with default values as in the original version from

You should see in the terminal :

```python
In [3]: top.optimize()
it.: 1 , obj.: 59.825 Vol.: 0.500, ch.: 0.200
it.: 2 , obj.: 24.137 Vol.: 0.500, ch.: 0.200
...
it.: 84 , obj.: 8.464 Vol.: 0.500, ch.: 0.001
it.: 85 , obj.: 8.464 Vol.: 0.500, ch.: 0.001
Elapsed time : 4.980943202972412 s

```

If you need to plot the evolution of the optimization then, you need to specify

```python
top.optimize(store=True)
```

This will store all density values for the whole optimization and allow to plot a animation of the optimization and the optimization objective at once.


## Features

OO version allow to seamlessy force change in dependent parameters. Typically when changing `nx` (number of elements in `x` direction), `ndofs` (number of degrees of freedom) change as well. To see it :

```python
In [4]: top
Out[7]:
Topology optimization
   50 elements in x_direction, 50 elements in y direction
   5202 total number of degrees of freedom   0.5 of total volume allowed
   5.4 radius filter
```

and try to change `nx`

```python
In [8]: top.nx = 100
Caution this will change number of dofs and hence
 the optimization problem

In [9]: top
Out[9]:
Topology optimization
   100 elements in x_direction, 50 elements in y direction
   10302 total number of degrees of freedom   0.5 of total volume allowed
   5.4 radius filter
```
