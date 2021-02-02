from SetTopol_multiple_loads import TopolSettings
import numpy as np
import random
from multiprocessing import Pool,RLock
import datetime
import os
import json
import cv2

from scipy.stats.distributions import norm, uniform, bernoulli, poisson
from pyDOE import *

file_lock=RLock()

def generate_design(params):

    count = 0
    top = TopolSettings(nx = params['nx'], ny = params['ny'], nbr_loads = params['nbr_loads'], vol = params['volfraction'], rmin = params['rmin'], penalinit = 3.0, penalmed = 3.0, filt = params['filt'], nu = 0.3)
    possible_fixed_nodes = np.arange(0, top.ny+1).tolist()+ [m*(top.ny +1)-1 for m in range(2,top.nx+1)] + np.sort(np.arange((top.ny+1)*top.nx, (top.nx+1)*(top.ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(top.ny+1) for m in range(1,top.nx)])).tolist()[::-1]  
    
    fixed_nodes = params['fixed_nodes']
    nodes = params['load_nodes']

    if (len(set(nodes) - set(fixed_nodes)) == len(set(nodes))) : # i do not want to put load on a fixed edge    
        top.setf(values=np.ones(params['nbr_loads']).tolist(), nodes=nodes, tetas=params['teta'])
        top.fixed = fixed_nodes
        top.optimize_with_DL_constraint()
        

    return "successfull"
                    
                    
if __name__ == "__main__" :

    print(os.cpu_count())

    volfractions = norm(loc=0.3, scale=0.05).ppf(lhs(50, samples=1)).reshape(50,)#norm(loc=0.3, scale=0.1).ppf(lhs(50, samples=1)).reshape(50,)  # this gives the x values having y-values equal to volume_fraction
    # rmins_1 = uniform(1.1, 2).ppf(lhs(50, samples=1)).reshape(50,) #  0.1 < volume fraction < 0.4 => 1.1 < Rmin < 3 ; Rmin suit la loi uniforme (1.1, 3 )
    # rmins_2 = uniform(2, 3).ppf(lhs(50, samples=1)).reshape(50,) #  0.4 < volume fraction < 0.6 => 2 < Rmin < 3 ; Rmin suit la loi uniforme (2, 3 )
    # rmins_3 = uniform(3, 4).ppf(lhs(50, samples=1)).reshape(50,) # volume fraction > 0.6 => 3 < Rmin < 4; Rmin suit la loi uniforme (3, 4 )
    filters = bernoulli(0.5).ppf(lhs(50, samples=1)).reshape(50,) # either present (1) or absent (0)
    tetas = uniform(0, 180).ppf(lhs(100, samples=1)).reshape(100,).tolist()#uniform(0, 60).ppf(lhs(30, samples=1)).reshape(30,).tolist()+  uniform(60, 130).ppf(lhs(30, samples=1)).reshape(30,).tolist() + uniform(130, 180).ppf(lhs(30, samples=1)).reshape(30,).tolist() 
    nbr_loads = poisson(2).ppf(lhs(50, samples=1)).reshape(50,) # most probable nbr_loads is 2
    windows = poisson(100).ppf(lhs(50, samples=1)).reshape(50,)
    nx = 100
    ny = 100
    window = int(nx/2) #
    possible_fixed_nodes = np.arange(0, ny+1).tolist()+ [m*(ny +1)-1 for m in range(2,nx+1)] + np.sort(np.arange((ny+1)*nx, (nx+1)*(ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(ny+1) for m in range(1,nx)])).tolist()[::-1]  


    total_nbr_samples = 100
    params_list = []
    for cnt in range(total_nbr_samples):
        volfraction = random.choice(volfractions)
        rmin = 2.4 # after the first generation phase, we have concluded that rmin should be less than 3 and in the range 2 to 2.8
        # if volfraction<0.4:
        #     rmin = random.choice(rmins_1)
        # elif volfraction>=0.4 and volfraction<0.6:
        #     rmin = random.choice(rmins_2)
        # else:
        #     rmin = random.choice(rmins_3)
        filt = random.choice(filters)
        the_nbr_loads = int(random.choice(nbr_loads))
        window = int(random.choice(windows))
        if the_nbr_loads >0:
            teta = random.sample(tetas, the_nbr_loads)
            for i in  range(0, len(possible_fixed_nodes), int(nx/2)):
                # fixed nodes
                fixed_nodes = possible_fixed_nodes[i:i+window]
                if(len(fixed_nodes) >1):
                    # opposite side nodes (load nodes)
                    # [opposite_side_nodes[0],opposite_side_nodes[params['window']],opposite_side_nodes[-1] ] here we take 3 opposite side nodes equidistant: 
                    # the 1st node is the edge node, the 2nd at distance = window from the 1st chosen node, 
                    # the 3rd node at distance = window from the 2nd chosen node and 2*window from the 1st chosen node 
                    opposite_side_nodes = []
                    element = fixed_nodes[(len(fixed_nodes) - 1 )]
                    starting_index = (possible_fixed_nodes.index(element)+nx)%len(possible_fixed_nodes)
                    ending_index = (possible_fixed_nodes.index(element)+2*nx)%len(possible_fixed_nodes)+1
                    opposite_side_nodes = []
                    if ending_index<starting_index:
                        opposite_side_nodes = possible_fixed_nodes[starting_index: ] + possible_fixed_nodes[0:ending_index]
                    else:
                        opposite_side_nodes = possible_fixed_nodes[starting_index: ending_index]
                    indices_load_nodes = np.linspace(0, len(opposite_side_nodes)-1, int(the_nbr_loads), dtype=int)
                    opposite_side_nodes = [opposite_side_nodes[i] for i in indices_load_nodes]#random.sample(opposite_side_nodes, int(the_nbr_loads))
                    params_list.append( {'nx':nx, 'ny':ny, 'volfraction':volfraction, 'rmin':rmin, 'filt':filt, 'fixed_nodes':fixed_nodes,'nbr_loads':the_nbr_loads, 'load_nodes':opposite_side_nodes, 'teta':teta, 'window': window} )

    pool = Pool(processes = 1)#os.cpu_count()) 
    result_df = pool.map(generate_design, params_list)
    if len(result_df)>0:
        print("successful generation")
