xs = [1,2,3]
from typing import List
import numpy as np 


def kr(edofmat):
    np.kron(edofmat,np.ones((8,1))).flatten()




def process(xs: List[int]) -> None:

