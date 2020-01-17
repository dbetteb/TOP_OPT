'''
  SetTopol.py
'''

import numpy as np
from numba import jit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from ElemMatrix import lk
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from toto import kr

class TopolSettings(object):
	OC_ITER = 60

	def __init__(self, nx = 50, ny = 50, nbr_loads = 1, vol = 0.5, rmin = 5.4, penalinit = 3.0, penalmed = 3.0, filt = 0, nu=0.3):
		self.__nx    = nx
		self.__ny    = ny
		self.__vol   = float(vol)
		self.__rmin  = rmin
		self.__penalinit = penalinit
		self.__penalmed  = penalmed
		self.__filt  = filt
		self.__nu    = nu

		self.__ndof = 2*(nx+1)*(ny+1)
		self.__xinit = float(vol)*np.ones(ny*nx)
		self.Emin = 1e-9
		self.Emax = 1.

		self.__edofmat, self.__iK, self.__jK = createedofmat(nx, ny)
		self.__edofmat = self.__edofmat.astype(int)
		self.__iH, self.__jH, self.__sH = createiHjHsH(nx, ny, rmin)
		self.__H = coo_matrix((self.__sH,(self.__iH, self.__jH)), shape=(nx*ny, nx*ny)).tocsc()
		self.__Hs = self.__H.sum(1)
		self.__g = 0.

		self.__number_of_loads = nbr_loads
		self.__u = np.zeros((self.__ndof, self.__number_of_loads))
		self.__f = np.zeros((self.__ndof, self.__number_of_loads))
		#################### f loads properties #####################
		self.__load_nodes = [0 for i in range(self.__number_of_loads)]
		self.__tetas = [0.0 for i in range(self.__number_of_loads)]
		self.__valuefs = [0.0 for i in range(self.__number_of_loads)]
		##############################################################
		
		################# fixed nodes properties #####################
		self.__fixed = np.arange(self.__ndof)
		self.__list_fixed_nodes = []
		self.__fixed_part = ""
		##############################################################

		self.__free = None

		# self.__free, self.__f, self.__u = createBCsupport(nx, ny, self.__ndof, self.__number_of_loads)[1:]

		self.__comphist = []

	def __repr__(self):
		st =f"Topology optimization \n" \
		f"   {self.nx} elements in x_direction, {self.ny} elements in y direction \n" \
		f"   {self.ndof} total number of degrees of freedom" \
		f"   {self.vol} of total volume allowed\n" \
		f"   {self.rmin} radius filter"
		return st


	def __getnu(self):
		return self.__nu

	def __setnu(self, value):
		print("Poisson coeficient should be between -0.5 and 1.")
		if value < -0.5:
			self.__nu = -0.5
		elif value > 1.:
			self.__nu = 1.
		else:
			self.__nu = value

	nu = property(__getnu, __setnu)

	def __getpenalinit(self):
		return self.__penalinit

	def __setpenalinit(self, value):
		print('Penalization power for SIMP should be between 3 and 6')
		if value < 3.:
			self.__penalinit = 3.
		elif value > 6.:
			self.__penalinit = 6.
		else:
			self.__penalinit = value

	penalinit = property(__getpenalinit, __setpenalinit)

	def __getpenalmed(self):
		return self.__penalmed

	def __setpenalmed(self, value):
		print('Penalization power for SIMP should be between 3 and 6')
		if value < 3.:
			self.__penalmed = 3.
		elif value > 6.:
			self.__penalmed = 6.
		else:
			self.__penalmed = value

	penalmed = property(__getpenalmed, __setpenalmed)



	def __getfilt(self):
		return self.__filt

	def __setfilt(self, value):
		print('To be tuned to be changed')
		if (value != 0) & (value !=1):
			self.__filt = 0
		else:
			self.__filt = value

	filt = property(__getfilt, __setfilt)

	def __getg(self):
		return self.__g

	def __setg(self, value):
		self.__g = float(value)

	g = property(__getg, __setg)


	def __getnx(self):
		return self.__nx

	def __setnx(self, value):
		print("Caution this will change number of dofs and hence \n the optimization problem")
		self.__nx = int(value)
		self.__xinit = float(self.vol)*np.ones(int(value)*self.__ny)
		self.__ndof = 2*(int(value)+1)*(self.__ny+1)
		self.__edofmat, self.__iK, self.__jK = createedofmat(int(value), self.__ny)
		self.__edofmat = self.__edofmat.astype(int)
		self.__iH, self.__jH, self.__sH = createiHjHsH(int(value), self.__ny, self.rmin)
		self.__H = coo_matrix((self.__sH,(self.__iH, self.__jH)), shape=(int(value)*self.__ny, int(value)*self.__ny)).tocsc()
		self.__Hs = self.__H.sum(1)
		self.__free, self.__f, self.__u = createBCsupport(int(value), self.__ny, self.__ndof, self.__number_of_loads)[1:]

	nx = property(__getnx, __setnx)


	def __getny(self):
		return self.__ny

	def __setny(self, value):
		print("Caution this will change number of dofs and hence \n the optimization problem")
		self.__ny = int(value)
		self.__xinit = float(self.vol)*np.ones(self.__nx*int(value))
		self.__ndof = 2*(self.__nx+1)*(int(value)+1)
		self.__edofmat, self.__iK, self.__jK = createedofmat(self.__nx, int(value))
		self.__edofmat = self.__edofmat.astype(int)
		self.__iH, self.__jH, self.__sH = createiHjHsH(self.__nx, int(value), self.rmin)
		self.__H = coo_matrix((self.__sH,(self.__iH, self.__jH)), shape=(self.__nx*int(value), self.__nx*int(value))).tocsc()
		self.__Hs = self.__H.sum(1)
		self.__free, self.__f, self.__u = createBCsupport(self.__nx, int(value), self.__ndof, self.__number_of_loads)[1:]

	ny = property(__getny, __setny)

	def __getrmin(self):
		return self.__rmin

	def __setrmin(self, value):
		print("Caution this will change filter connectivity matrices hence \n the optimization problem")
		self.__rmin = float(value)
		self.__iH, self.__jH, self.__sH = createiHjHsH(self.__nx, self.__ny, self.__rmin)
		self.__H = coo_matrix((self.__sH,(self.__iH, self.__jH)), shape=(self.__nx*self.__ny, self.nx*self.__ny)).tocsc()
		self.__Hs = self.__H.sum(1)

	rmin = property(__getrmin, __setrmin)

	def __getedofmat(self):
		return self.__edofmat

	def __setedofmat(self, value):
		print("This cannot be changed")

	edofmat = property(__getedofmat, __setedofmat)

	def __getiK(self):
		return self.__iK

	def __setiK(self, value):
		print("This cannot be changed")

	iK = property(__getiK, __setiK)

	def __getjK(self):
		return self.__jK

	def __setjK(self, value):
		print("This cannot be changed")

	jK = property(__getjK, __setjK)

	def __getiH(self):
		return self.__iH

	def __setiH(self, value):
		print("This cannot be changed")

	iH = property(__getiH, __setiH)

	def __getjH(self):
		return self.__jH

	def __setjH(self, value):
		print("This cannot be changed")

	jH = property(__getjH, __setjH)

	def __getsH(self):
		return self.__sH

	def __setsH(self, value):
		print("This cannot be changed")

	sH = property(__getsH, __setsH)

	def __getH(self):
		return self.__H

	def __setH(self, value):
		print("This cannot be changed")

	H = property(__getH, __setH)


	def __getHs(self):
		return self.__Hs

	def __setHs(self, value):
		print("This cannot be changed")

	Hs = property(__getHs, __setHs)



	def __getxinit(self):
		return self.__xinit

	def __setxinit(self, value):
		print("initialization will change automatically when" + \
				"vol, nx or ny is changed")

	xinit = property(__getxinit, __setxinit)

	def __getvol(self):
		return self.__vol

	def __setvol(self, value):
		if value < 0:
			print("Volume fraction should be greater than 0, set to default 0.5")
			self.__vol = float(0.5)
			self.__xinit = float(0.5)*np.ones(self.__nx*self.__ny)
		elif value > 1:
			print("Volume fraction should be less than 1, set to default 0.5")
			self.__vol = float(0.5)
			self.__xinit = float(0.5)*np.ones(self.__nx*self.__ny)
		else:
			self.__vol = float(value)
			self.__xinit = float(value)*np.ones(self.__nx*self.__ny)

	vol = property(__getvol, __setvol)

	def __getndof(self):
		return self.__ndof

	def __setndof(self, value):
		print("Number of dof cannot be changed")

	ndof = property(__getndof, __setndof)

	def __getfree(self):
		return self.__free

	def __setfree(self, value):
		print("Cannot be changed")

	free = property(__getfree, __setfree)

	def __getnumber_of_loads(self):
		return self.__number_of_loads

	def __setnumber_of_loads(self, value):
		if (value>0 and value<self.__ndof):
			self.__number_of_loads = value
		else:
			print("Invalid Number of Loads. For a "+str(self.nx)+" x "+str(self.ny)+" volume, You can have 0 < Number_Of_Loads < "+ str(self.ndof))
			print("Number_Of_Loads will be set to 1")
			self.__number_of_loads = 1

	number_of_loads = property(__getnumber_of_loads, __setnumber_of_loads)

	def getf(self):
		return self.__f

	def setf(self, values, nodes=[0], tetas=[0.0]):
		"""
		Sets the loads vector according to the position (node), orientation (teta) and intensity (value) chosen by the user

		Parameters
		----------
		values
			List[float] : list of load intensities in Newton
		nodes
			List[int] : list of the node numbers where we want to apply each load; the nodes are numbered column-wise from the left of the rectangle to its right from 0 to (__nx+1)*(__ny+1) - 1
		tetas
			List[float] : list of load orientations in degrees

		NB: 
		__ndof = 2*(__nx+1)*(__ny +1)
		in a __nx*__ny volume, there are (__nx+1)*(__ny+1) nodes each having 2 degrees of freedom the horizontal
		and the vertical displacements
		Thus, __f is = [[Fx0], [Fy0], [Fx1], [Fy1], ..., [Fxn], [Fyn]] such that n = (1+__nx)*(1+__ny) - 1
		and Fx = load intensity along the x-axis = value*cos(teta)
		and Fy = load intensity along the y-axis = value*sin(teta)

		if node=(__ny+1)*__nx => we have chosen the upper right node to put the load on => position of Fx = x_pos = __ndof - 2*(__ny+1) - 1 = 2*__nx*(__ny+1) - 1 and position of Fy = y_pos = position of Fx + 1
		if node = 0 => we have chosen the upper left node to put the load on => position of Fx = 0 and position of Fy = 1
		
		####################################
		############# QUESTION #############
		####################################
		DOES F FOLLOW THE SAME CONSTRAINTS AS THE FIXED NODES I.E. COULD WE ONLY APPLY F ON EDGE NODES OR EVERYWHERE?

		"""
		f = np.zeros((self.ndof, self.__number_of_loads))
		for i in range(self.__number_of_loads):
			node = nodes[i]
			value = values[i]
			teta = tetas[i]
			if (node < 0) or (node > (self.nx+1)*(self.ny+1) - 1):
				print("Invalid node number "+ str(node), ". Node number should be >0 and <="+ str((self.nx+1)*(self.ny+1) - 1))
				print("The load will be set on the 1st node: N0")
				node = 0
			
			x_pos = 2*node
			y_pos = x_pos+1

			Fx = value*np.cos(teta*np.pi/180)
			Fy = value*np.sin(teta*np.pi/180)
			
			
			f[x_pos, i] = Fx
			f[y_pos, i] = Fy
		
		self.__load_nodes = nodes
		self.__tetas = tetas
		self.__valuefs = values

		self.__f = f


	f = property(getf, setf)

	def __getload_nodes(self):
		return self.__load_nodes

	def __setload_nodes(self, value):
		print("It is set by the load setter.")

	load_nodes = property( __getload_nodes, __setload_nodes)


	def __gettetas(self):
		return self.__tetas

	def __settetas(self, value):
		print("It is set by the load setter.")

	tetas = property(__gettetas, __settetas)

	
	def __getvaluefs(self):
		return self.__valuefs

	def __setvaluefs(self, value):
		print("It is set by the load setter.")

	valuefs = property( __getvaluefs, __setvaluefs)

	def __getfixed(self):
		return self.__fixed

	def __setfixed(self, list_nodes):
		""" 
		Sets the fixed nodes chosen by the user
		NB: this function triggers simultaneously self.__free

		Parameters
		----------
		list_nodes
			List[int] : the list of node numbers where we want to fix the object; the nodes are numbered column-wise from the left of the rectangle to its right from 0 to (__nx+1)*(__ny+1) - 1

		NB: we can only fix edge nodes i.e. contour nodes of the rectangle, here called possible_fixed_nodes
		one example a nx = 4, ny = 2 rectangle has as edge nodes [0,1,2,3,5,6,8,9,11,12,13,14]

		TIPS:
		fixed_nodes_numbers = np.arange(0,ny+1).tolist()==================================> all the left side is fixed
		fixed_nodes_numbers = [m*(ny+1) for m in range(0,nx+1)]===========================> all the upper side is fixed
		fixed_nodes_numbers = [m*(ny +1)-1 for m in range(1,nx+2)]]=======================> all the bottom side is fixed
		fixed_nodes_numbers = np.arange((ny+1)*nx, (nx+1)*(ny+1)).tolist()================> all the right side is fixed

		The fixed nodes in the list possible_fixed_nodes follow an anti-clockwise direction i.e. if nx = 4, ny = 2 
		the edge nodes list is [0,1,2,5,8,11,14,13,12,9,6,3]
		
		"""
		possible_fixed_nodes = np.arange(0, self.ny+1).tolist()+ [m*(self.ny +1)-1 for m in range(2,self.nx+1)] + np.sort(np.arange((self.ny+1)*self.nx, (self.nx+1)*(self.ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(self.ny+1) for m in range(1,self.nx)])).tolist()[::-1]  
		if (len(set(list_nodes)-set(possible_fixed_nodes))>0):
			print("Invalid node numbers "+ str(set(list_nodes)-set(possible_fixed_nodes)), ". A fixed Node can only be one of the following list :"+ str(possible_fixed_nodes))
			print("The load will be set on the 1st node: N0")
			node = 0
		
		dofs = np.arange(self.ndof)
		fixed = []
		for node in list_nodes:
			x_pos = 2*node
			y_pos = x_pos+1
			fixed = fixed + [x_pos, y_pos]

		free  = np.setdiff1d(dofs, fixed)

		self.__list_fixed_nodes = list_nodes
		self.__fixed = fixed
		self.__free = free 

	fixed = property(__getfixed, __setfixed)

	def __getlist_fixed_nodes(self):
		return self.__list_fixed_nodes

	def __setlist_fixed_nodes(self, value):
		print("It is set by the fixed setter.")


	list_fixed_nodes = property( __getlist_fixed_nodes, __setlist_fixed_nodes)

	def __getfixed_part(self):
		return self.__fixed_part 

	def __setfixed_part(self, value):

		if (len(set(self.list_fixed_nodes)- set(np.arange(0,self.ny+1).tolist()))==0):
			self.__fixed_part = "left_side"
		elif (len(set(self.list_fixed_nodes)- set([m*(self.ny+1) for m in range(0,self.nx+1)]))==0):
			self.__fixed_part = "upper_side"
		elif (len(set(self.list_fixed_nodes)-set([m*(self.ny+1)-1 for m in range(1,self.nx+2)]))==0):
			self.__fixed_part = "bottom_side"
		elif (len(set(self.list_fixed_nodes)-set(np.arange((self.ny+1)*self.nx, (self.nx+1)*(self.ny+1)).tolist()))==0):
			self.__fixed_part = "right_side"
		else:
			self.__fixed_part = "complex"

	fixed_part = property(__getfixed_part, __setfixed_part )

	def __setu(self, value):
		print("Cannot be changed")

	def __getu(self):
		return self.__u

	u = property(__getu, __setu)


	def optimize(self, changecriteria = 1e-3, maxiter = 100, store=False, cond = False, loop_switch = 20):
		"""
		   main optimizer
		"""
		tstart = time.time()
		loop, change = 0, 1
		nx, ny = self.__nx, self.__ny
		u = self.u
		f = self.f
		dv = np.ones(nx*ny)
		dc = np.ones(nx*ny)
		ce = np.ones(nx*ny)
		KE = lk(E = self.Emax, nu = self.nu).create_matrix()
		x     = self.xinit.copy()
		xold  = self.xinit.copy()
		xphys = self.xinit.copy()
		g = 0
		comp = []
		if store:
			hi = []
			hi.append(self.xinit.copy())
		while (change > changecriteria) and (loop < maxiter):
			loop += 1
			if loop < loop_switch:
				penal = self.penalinit
			else:
				penal = self.penalmed
			sK=((KE.flatten()[np.newaxis]).T*(self.Emin+(xphys)**penal*(self.Emax-self.Emin))).flatten(order='F')
			K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
            # remove constrained dofs
			K = K[self.free,:][:,self.free]
			if cond:
				print(K)
				cd = []
				cd.append(np.linalg.cond(K))
			
			########## multiple loads ############
			dc = np.zeros(nx*ny)
			obj = 0.0
			for i in range(self.__number_of_loads):
				u[self.free,i] = spsolve(K, f[self.free,i])
				ce[:] = (np.dot(u[self.edofmat, i].reshape(nx*ny,8),KE) * u[self.edofmat, i].reshape(nx*ny,8) ).sum(1)
				obj   = obj + ( (self.Emin+xphys**penal*(self.Emax-self.Emin))*ce ).sum()
				dc[:] = dc[:] + (-penal*xphys**(penal-1.)*(self.Emax-self.Emin))*ce
			comp.append(obj)
			########## multiple loads ############
			dv[:] = np.ones(ny*nx)
			if self.filt == 0: # mesh-independency filter method
				dc[:] = np.asarray((self.H*(x*dc))[np.newaxis].T/self.Hs)[:,0] / np.maximum(0.001, x)
			elif self.filt == 1:
				dc[:] = np.asarray(self.H*(dc[np.newaxis].T/self.Hs))[:,0]
				dv[:] = np.asarray(self.H*(dv[np.newaxis].T/self.Hs))[:,0]
			#print(dc)
			xold[:] = x
			x[:], g = oc(nx, ny, x, self.vol, dc, dv, g, TopolSettings.OC_ITER)
			if self.filt == 0:
				xphys[:] = x
			elif self.filt == 1:
				xphys[:]=np.asarray(self.H*x[np.newaxis].T/self.Hs)[:,0]
			change=np.linalg.norm(x.reshape(nx*ny,1)-xold.reshape(nx*ny,1),np.inf)
			if store:
				hi.append(xphys.copy())
			print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
					loop,obj,(g+self.vol*nx*ny)/(nx*ny),change))
			if loop == maxiter:
				self.finalcomp = ( (self.Emin+xphys*(self.Emax-self.Emin))*ce ).sum()
		telap = time.time()-tstart
		print(f"Elapsed time : {telap} s")
		self.__comphist = comp # list of objective function values
		self.res = xphys
		if store:
			self.hist = hi
		if cond:
			self.cond = cd

	def __getcomphist(self):
		return self.__comphist

	def __setcomphist(self, value):
		print("Historical internal data cannot be set")

	comphist = property(__getcomphist, __setcomphist)


	def plot(self, name='test_plot'):
		if not hasattr(self, 'hist'):
			print('No stored data, please re run topology optimization with store=True')
		else:
			fig, (ax1,ax2) = plt.subplots(ncols=2)
			ims = []
			for i in range(len(self.comphist)):
				im1 = ax1.imshow(1.-self.hist[i].reshape(self.nx,self.ny).T, animated=True, cmap=plt.get_cmap('gray'), vmin=0., vmax =1.)
				im2, = ax2.plot(range(len(self.comphist)),self.comphist,'b',lw=3)
				im2, = ax2.plot(i, self.comphist[i],'ro',markersize=8)
				asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
				asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
				ax2.set_aspect(asp)
				ims.append([im1,im2])
			animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=400)

			print("Saving plot ... ")
			

			fig.savefig('./data/'+name)
			# fig.savefig('./data/volfrac_'+str(self.__vol)+'_rmin_'+str(self.__rmin)+'_ft_'+str(self.__filt)+'_load_of_intensities_'+str(self.valuefs)+'_orientations_'+str(self.tetas)+'_on_node_number_'+str(self.load_nodes)+'_fixed_on_nodes'+str(self.fixed_part)+'.jpeg')
		return fig


@jit(nopython=True)
def oc(nx, ny, x, volfrac, dc, dv, g, oc_iter):
	l1, l2 = 0., 1e9
	move = 0.2
	xnew = np.zeros(nx*ny)
	for i in range(oc_iter):
	#while (l2-l1)/(l2+l1) > 1e-3:
		lmid = .5*(l2+l1)
		xnew[:] = np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
		gt = g + np.sum((dv*(xnew-x)))
		if gt > 0:
			l1 = lmid
		else:
			l2 = lmid
	return xnew, gt



@jit(nopython=True)
def createedofmat(nx, ny):
	"""
	   returns edofMat, iK, jK
	"""
	edofMat = np.zeros((nx*ny, 8))
	for elx in range(nx):
		for ely in range(ny):
			el = ely + elx*ny
			n1 = (ny+1)*elx+ely
			n2 = (ny+1)*(elx+1)+ely
			edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
	return edofMat, np.kron(edofMat,np.ones((8,1))).flatten(), np.kron(edofMat,np.ones((1,8))).flatten()# kr(edofMat, 8,1), kr(edofMat, 1,8), np.kron(edofMat,np.ones((8,1))).flatten()


@jit(nopython=True)
def createiHjHsH(nx, ny, rmin):
	"""
        returns iH, jH, sH
	"""
	nfilter=int(nx*ny*((2*(np.ceil(rmin)-1)+1)**2))
	#nfilter=nfilter.astype(int)
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
	cc=0
	for i in range(nx):
		for j in range(ny):
			row=i*ny+j
			kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
			kk2=int(np.minimum(i+np.ceil(rmin),nx))
			ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
			ll2=int(np.minimum(j+np.ceil(rmin),ny))
			for k in range(kk1,kk2):
				for l in range(ll1,ll2):
					col=k*ny+l
					fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
					iH[cc]=row
					jH[cc]=col
					sH[cc]=np.maximum(0.0,fac)
					cc=cc+1
	return iH, jH, sH


#@jit(nopython=True)
def createBCsupport(nx, ny, ndofs, load_nbr = 1, BCtype = "cant"):
	"""
	    creates BC, support, RHS and initialization
    """
	if BCtype == "MBB":
		dofs  = np.arange(ndofs)
		fixed = np.union1d(dofs[0:2*(ny+1):2], np.array([2*(nx+1)*(ny+1)-1]))
		free  = np.setdiff1d(dofs, fixed)
		f, u = np.zeros((ndofs, load_nbr)), np.zeros((ndofs, load_nbr))
		for i in range(load_nbr):
			f[:-1, i]=-1
	else:
		dofs  = np.arange(ndofs)
		fixed = dofs[0:2*(ny+1)]
		free  = np.setdiff1d(dofs, fixed)
		f, u = np.zeros((ndofs, load_nbr)), np.zeros((ndofs, load_nbr))
		for i in range(load_nbr):
			f[ndofs-2*ny,i] = -1
	return fixed, free, f, u
