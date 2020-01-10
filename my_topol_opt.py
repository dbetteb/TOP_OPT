'''
  MyTopolOpt.py
  '''
  
import argparse
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors 
import matplotlib.pyplot as plt


from ElemMatrix import lk

parser = argparse.ArgumentParser(prog = 'my_topol_opt.py', \
    description = 'Topology optimization in Python', \
    epilog = 'Author : Dimitri Bettebghor \n'  + \
              'to do list : many things \n' + \
              'finish the basis program \n'
              , \
    add_help = True)

parser.add_argument('nx', metavar='nx', type= int, nargs = 1,\
    help='Number of elements along the x direction (positive integer)')
parser.add_argument('ny', metavar='ny', type= int, nargs = 1,\
    help='Number of elements along the y direction (positive integer)')
parser.add_argument('vol', metavar='vol', type = float, nargs = 1, \
    help='Fraction allowed for matter ditribution')

args = parser.parse_args()

nx, ny = args.nx[0], args.ny[0]
vol    = args.vol[0]


