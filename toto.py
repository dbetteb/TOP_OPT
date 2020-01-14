xs = [1,2,3]
from typing import List
import numpy as np 


def kr(edofmat, nx, ny):
    return np.kron(edofmat,np.ones((nx,ny))).flatten()




def process(xs: List[int]) -> None:
	return

