from SetTopol_multiple_loads import TopolSettings
import numpy as np
import random
from multiprocessing import Pool,RLock
import datetime
import os
import json

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
        top.optimize(store=True)
        # Plot and Save only and only if the OBJECTIVE FUNCTION VALUE decreased i.e. the optimization problem converged
        if (top.comphist[0]> top.comphist[-1]): #(top.comphist[0]/ top.comphist[-1] >= 5):
            datetime_info = str(datetime.datetime.now())[0:23].replace(':', '_').replace('.','_')
            top.plot(name = datetime_info)
            # d1 is the distance between node0 and the 1st fixed node
            # d1 = index of 1st fixed node in the possible_fixed_nodes list / perimeter of the volume such that the perimeter of the volume = 2*(nx+ny) = len(possible_fixed_nodes)
            d1 = possible_fixed_nodes.index(top.list_fixed_nodes[0])/len(possible_fixed_nodes)
            # d2 is the distance between node0 and the last fixed node
            # d2 = index of lasst fixed node in the possible_fixed_nodes list / perimeter of the volume such that the perimeter of the volume = 2*(nx+ny) = len(possible_fixed_nodes)
            d2 = possible_fixed_nodes.index(top.list_fixed_nodes[-1])/len(possible_fixed_nodes)
            # d3 is the distance between node0 and the load node
            # NB: a load node here is an edge node
            # d3 = index of load node in the possible_fixed_nodes list / perimeter of the volume such that the perimeter of the volume = 2*(nx+ny) = len(possible_fixed_nodes)
            # Since there could be N load nodes, d3 is a list of distances
            d3 = [possible_fixed_nodes.index(l)/len(possible_fixed_nodes) for l in top.load_nodes ]
            # d1, d2 and d3 are now borned parameters => GAN can easily learn to recreate them

            data = {'nx':top.nx, 'ny':top.ny, 'volume_fraction':top.vol, 'filter_rmin': top.rmin, 
                    'initial_penalty':top.penalinit, 'med_penalty': top.penalmed, 
                    'mesh_independency_filter':top.filt,'poisson_ratio_nu':top.nu, 
                    'number_of_nodes':top.ndof, 'Young_modulus_Emin':top.Emin, 
                    'Young_modulus_Emax':top.Emax, 'load_nodes':top.load_nodes, 
                    'load_orientations':top.tetas, 'load_intensities':top.valuefs,
                    'fixed_nodes':top.list_fixed_nodes, 'type_fixed_nodes':top.fixed_part, 
                    'distanceBetween_node0_1st_fixed_node': d1, 'distanceBetween_node0_last_fixed_node':d2, 'distanceBetween_node0_load_nodes':d3,
                    'final_objective_function_value':top.comphist[-1], 
                    'all_objective_function_values':top.comphist, 
                    'design_reference':datetime_info, 'time_for_convergence': top.time_required}

            with file_lock:
                try:
                    print("Writing to json file ...")
                    with open('./parameters/sampler_design_generation_'+str(datetime.datetime.now())[0:10]+'.json') as f:
                        data_old = json.load(f)

                    data_old.append(data)
                except:
                    data_old= [data]

                with open('./parameters/sampler_design_generation_'+str(datetime.datetime.now())[0:10]+'.json', 'w') as f:
                    json.dump(data_old, f)

    return "successfull"
                    
                    
if __name__ == "__main__" :

    print(os.cpu_count())

    volfractions = norm(loc=0.3, scale=0.1).ppf(lhs(50, samples=1)).reshape(50,)  # this gives the x values having y-values equal to volume_fraction
    # rmins_1 = uniform(1.1, 2).ppf(lhs(50, samples=1)).reshape(50,) #  0.1 < volume fraction < 0.4 => 1.1 < Rmin < 3 ; Rmin suit la loi uniforme (1.1, 3 )
    # rmins_2 = uniform(2, 3).ppf(lhs(50, samples=1)).reshape(50,) #  0.4 < volume fraction < 0.6 => 2 < Rmin < 3 ; Rmin suit la loi uniforme (2, 3 )
    # rmins_3 = uniform(3, 4).ppf(lhs(50, samples=1)).reshape(50,) # volume fraction > 0.6 => 3 < Rmin < 4; Rmin suit la loi uniforme (3, 4 )
    filters = bernoulli(0.5).ppf(lhs(50, samples=1)).reshape(50,) # either present (1) or absent (0)
    tetas = uniform(0, 60).ppf(lhs(30, samples=1)).reshape(30,).tolist()+  uniform(60, 130).ppf(lhs(30, samples=1)).reshape(30,).tolist() + uniform(140, 270).ppf(lhs(30, samples=1)).reshape(30,).tolist() 
    nbr_loads = poisson(2).ppf(lhs(50, samples=1)).reshape(50,) # most probable nbr_loads is 2
    windows = poisson(50).ppf(lhs(50, samples=1)).reshape(50,)
    nx = 100
    ny = 100
    window = int(nx/2) #
    possible_fixed_nodes = np.arange(0, ny+1).tolist()+ [m*(ny +1)-1 for m in range(2,nx+1)] + np.sort(np.arange((ny+1)*nx, (nx+1)*(ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(ny+1) for m in range(1,nx)])).tolist()[::-1]  


    total_nbr_samples = 1
    params_list = []
    for cnt in range(total_nbr_samples):
        print("helllo")
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
                    
                    opposite_side_nodes = random.sample(opposite_side_nodes, int(the_nbr_loads))
                    params_list.append( {'nx':nx, 'ny':ny, 'volfraction':volfraction, 'rmin':rmin, 'filt':filt, 'fixed_nodes':fixed_nodes,'nbr_loads':the_nbr_loads, 'load_nodes':opposite_side_nodes, 'teta':teta, 'window': window} )

    pool = Pool(processes = os.cpu_count()-2) 
    result_df = pool.map(generate_design, params_list)
    if len(result_df)>0:
        print("successful generation")