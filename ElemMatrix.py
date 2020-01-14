"""
ElemMatrix.py contains elementary stiffness matrix
"""

import numpy as np


class lk(object):
	def __init__(self, E = 1, nu = 0.3):
		self.E = E
		self.nu = nu
		
	
	def create_matrix(self):
		"""
		This function returns the element stiffness matrix.
		Since the element stiffness matrix for solid material is the same for all elements, 
		thus this function is only called once in the main code KE=lk()
		Parameters
		----------
		Default params
		E : elastic Young's modulus
		nu : poisson ratio = - transversal strain/axial strain

		Returns
		-------
		np.array
			8x8 array containing the stiffness information
		"""
		E = self.E
		nu = self.nu
		k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
		K = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
		[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
		[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
		[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
		[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
		[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
		[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
		[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
		return K
	

