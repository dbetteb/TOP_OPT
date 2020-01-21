from SetTopol_multiple_loads import TopolSettings
import numpy as np
import random
from multiprocessing import Pool
import pandas as pd
import datetime
import os
print(os.cpu_count())


def one_load_generate_design(params):

    list_nx   = []
    list_ny    = []
    list_vol   = []
    list_rmin  = [] 
    list_penalinit = []
    list_penalmed  = [] 
    list_filt  = [] 
    list_nu    = [] 
    list_ndof = [] 
    list_Emin = [] 
    list_Emax = [] 
    list_f = [] 
    list_node_f = [] 
    list_teta_f = [] 
    list_valuef = [] 
    list_fixed_nodes = []
    list_type_fixed_nodes = []
    list_final_objective_function_value = []
    list_all_objective_function_values = []
    list_design_references = []
    list_time_required = []

    count = 0
    top = TopolSettings(nx = params['nx'], ny = params['ny'], nbr_loads = 1, vol = params['volfraction'], rmin = params['rmin'], penalinit = 3.0, penalmed = 3.0, filt = params['filt'], nu = 0.3)
    possible_fixed_nodes = np.arange(0, top.ny+1).tolist()+ [m*(top.ny +1)-1 for m in range(2,top.nx+1)] + np.sort(np.arange((top.ny+1)*top.nx, (top.nx+1)*(top.ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(top.ny+1) for m in range(1,top.nx)])).tolist()[::-1]  
    
    fixed_nodes = params['fixed_nodes']
    if len(fixed_nodes)>0:
        # opposite side nodes (load nodes)
        element = fixed_nodes[(len(fixed_nodes) - 1 )]
        starting_index = (possible_fixed_nodes.index(element)+top.nx)%len(possible_fixed_nodes)
        ending_index = (possible_fixed_nodes.index(element)+2*top.nx)%len(possible_fixed_nodes)+1
        opposite_side_nodes = []
        if ending_index<starting_index:
            opposite_side_nodes = possible_fixed_nodes[starting_index: ] + possible_fixed_nodes[0:ending_index]
        else:
            opposite_side_nodes = possible_fixed_nodes[starting_index: ending_index]

        for load_node in [opposite_side_nodes[0],opposite_side_nodes[params['window']],opposite_side_nodes[-1] ]:
            nodes = [load_node]
            if (len(set(nodes) - set(fixed_nodes)) == len(set(nodes))) : # i do not want to put load on a fixed edge    
                top.setf(values=[1], nodes=nodes, tetas=[params['teta']])
                top.fixed = fixed_nodes
                top.optimize(store=True)
                # Plot and Save only and only if the OBJECTIVE FUNCTION VALUE decreased i.e. the optimization problem converged
                if (top.comphist[0]> top.comphist[-1]): #(top.comphist[0]/ top.comphist[-1] >= 5):
                    datetime_info = str(datetime.datetime.now())[0:23].replace(':', '_').replace('.','_')
                    top.plot(name = datetime_info+'__'+str(count+1))
                    list_nx.append(top.nx)
                    list_ny.append(top.ny)
                    list_vol.append(top.vol)
                    list_rmin.append(top.rmin) 
                    list_penalinit.append(top.penalinit)
                    list_penalmed.append(top.penalmed) 
                    list_filt.append(top.filt) 
                    list_nu.append(top.nu) 
                    list_ndof.append(top.ndof) 
                    list_Emin.append(top.Emin) 
                    list_Emax.append(top.Emax) 
                    list_node_f.append(top.load_nodes) 
                    list_teta_f.append(top.tetas) 
                    list_valuef.append(top.valuefs) 
                    list_fixed_nodes.append(top.list_fixed_nodes)
                    list_type_fixed_nodes.append(top.fixed_part)
                    list_final_objective_function_value.append(top.comphist[-1])
                    list_all_objective_function_values.append(top.comphist)
                    list_design_references.append(datetime_info+'__'+str(count+1))
                    list_time_required.append(top.time_required)
                    count +=1
                    
        return pd.DataFrame(data = {'nx':list_nx, 'ny':list_ny, 'volume_fraction':list_vol, 'filter_rmin': list_rmin, 
                                        'initial_penalty':list_penalinit, 'med_penalty': list_penalmed, 
                                        'mesh_independency_filter':list_filt,'poisson_ratio_nu':list_nu, 
                                        'number_of_nodes':list_ndof, 'Young_modulus_Emin':list_Emin, 
                                        'Young_modulus_Emax':list_Emax, 'load_nodes':list_node_f, 
                                        'load_orientations':list_teta_f, 'load_intensities':list_valuef,
                                        'fixed_nodes':list_fixed_nodes, 'type_fixed_nodes':list_type_fixed_nodes, 
                                        'final_objective_function_value':list_final_objective_function_value, 
                                        'all_objective_function_values':list_all_objective_function_values, 
                                        'design_reference':list_design_references, 'time_for_convergence': list_time_required})
                    
                    
 
volfractions = [0.3,0.4, 0.5, 0.6]
rmins = [1.2, 2.7, 3.5, 4.3, 5.1]
filters = [0,1]
tetas = [0,45,90,180,180+45,180+90]
nx = 100
ny = 100
window = int(nx/2)
possible_fixed_nodes = np.arange(0, ny+1).tolist()+ [m*(ny +1)-1 for m in range(2,nx+1)] + np.sort(np.arange((ny+1)*nx, (nx+1)*(ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(ny+1) for m in range(1,nx)])).tolist()[::-1]  

pool = Pool(processes = os.cpu_count()-2) 
result_df = pool.map(one_load_generate_design, [{'nx':nx, 'ny':ny, 'volfraction':volfraction, 'rmin':rmin, 'filt':filt, 'fixed_nodes':possible_fixed_nodes[i:i+window], 'teta':teta, 'window': window} for volfraction in volfractions for rmin in rmins for filt in filters for i in  range(0, (ny +1 )*nx - window +1, int(nx/2)) for teta in tetas])

if len(result_df)>0:
    final_df = pd.concat(result_df)
    final_df.to_csv('./parameters/test_one_load_design_generation_'+str(datetime.datetime.now())[0:19].replace(':', '_')+'.csv')