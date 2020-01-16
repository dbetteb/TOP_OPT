###############################################################
#################### 2D Designs Generation ####################
####################      Waad ALMASRI     ####################
###############################################################

import numpy as np
import random
from SetTopol_multiple_loads import TopolSettings
import datetime
import pandas as pd



def one_load_design_generator(nx = 100, ny = 100, volfraction = 0.4, nbr_loads = 2, rmin = 1.2, penalinit = 3.0, penalmed = 3.0, filt = 0, nu=0.3 ):
	
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

	top = TopolSettings(nx = nx, ny = ny, nbr_loads = nbr_loads, vol = volfraction, rmin = 1.2, penalinit = penalinit, penalmed = penalmed, filt = filt, nu = nu)

	# all possible fixed edge nodes
	possible_fixed_nodes = np.arange(0, top.ny+1).tolist()+ [m*(top.ny +1)-1 for m in range(2,top.nx+1)] + np.sort(np.arange((top.ny+1)*top.nx, (top.nx+1)*(top.ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(top.ny+1) for m in range(1,top.nx)])).tolist()[::-1]  


	# left side nodes fixed
	list_fixed_nodes =  np.arange(0, top.ny+1).tolist()
	top.fixed=list_fixed_nodes

	# opposite side nodes
	element = list_fixed_nodes[(len(list_fixed_nodes) - 1 )]
	starting_index = (possible_fixed_nodes.index(element)+top.nx)%len(possible_fixed_nodes)
	ending_index = (possible_fixed_nodes.index(element)+2*top.nx)%len(possible_fixed_nodes)+1
	opposite_side_nodes = []
	if ending_index<starting_index:
	    opposite_side_nodes = possible_fixed_nodes[starting_index: ] + possible_fixed_nodes[0:ending_index]
	
	
	datetime_info = str(datetime.datetime.now())[0:19]
	plot_reference = 0

	print("Generating Designs ... ")

	for load_node in opposite_side_nodes:
		nodes = [load_node]
        for teta in [0,45,90,180,180+45,180+90]: # [45,90,180,180+45,180+90] with teta = 180+90 only we have 596 output_plot
            if len(set(nodes) - set(list_fixed_nodes)) == len(set(nodes)): # i do not want to put load on a fixed edge
                top.setf(values=[1], nodes=nodes, tetas=[teta])
                # top.fixed(list_nodes=list_fixed_nodes)
                top.optimize(store=True)
                top.plot(name = datetime_info+str(plot_reference+1))
                
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
				list_design_references.append(datetime_info+str(plot_reference+1))
                
                plot_reference+=1

    print("Saving Parameters ... ")
    df = pd.DataFrame(data = {'nx':list_nx, 'ny':list_ny, 'volume_fraction':list_vol, 'filter_rmin': list_rmin, 'initial_penalty':list_penalinit, 'med_penalty': list_penalmed, 'mesh_independency_filter':list_filt,
    	'poisson_ratio_nu':list_nu, 'number_of_nodes':list_ndof, 'Young_modulus_Emin':list_Emin, 'Young_modulus_Emax':list_Emax, 'load_nodes':list_node_f, 'load_orientations':list_teta_f, 'load_intensities':list_valuef,
    	'fixed_nodes':list_fixed_nodes, 'type_fixed_nodes':list_type_fixed_nodes, 'final_objective_function_value':list_final_objective_function_value, 'all_objective_function_values':list_all_objective_function_values, 'design_reference':list_design_references})
    df.to_excel('./parameters/one_load_design_generation'+datetime_info)


def two_load_design_generator(nx = 100, ny = 100, volfraction = 0.4, nbr_loads = 2, rmin = 1.2, penalinit = 3.0, penalmed = 3.0, filt = 0, nu=0.3 ):

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
	
	top = TopolSettings(nx = nx, ny = ny, nbr_loads = nbr_loads, vol = volfraction, rmin = 1.2, penalinit = penalinit, penalmed = penalmed, filt = filt, nu = nu)

	# all possible fixed edge nodes
	possible_fixed_nodes = np.arange(0, top.ny+1).tolist()+ [m*(top.ny +1)-1 for m in range(2,top.nx+1)] + np.sort(np.arange((top.ny+1)*top.nx, (top.nx+1)*(top.ny+1))).tolist()[::-1] + np.sort(np.asarray([m*(top.ny+1) for m in range(1,top.nx)])).tolist()[::-1]  


	# left side nodes fixed
	list_fixed_nodes =  np.arange(0, top.ny+1).tolist()
	top.fixed=list_fixed_nodes

	# opposite side nodes
	element = list_fixed_nodes[(len(list_fixed_nodes) - 1 )]
	starting_index = (possible_fixed_nodes.index(element)+top.nx)%len(possible_fixed_nodes)
	ending_index = (possible_fixed_nodes.index(element)+2*top.nx)%len(possible_fixed_nodes)+1
	opposite_side_nodes = []
	if ending_index<starting_index:
	    opposite_side_nodes = possible_fixed_nodes[starting_index: ] + possible_fixed_nodes[0:ending_index]

	datetime_info = str(datetime.datetime.now())[0:19]
	plot_reference = 0
	
	# 2 loads with similar orientations
	for nodes in random.sample(opposite_side_nodes,2):
        for teta in [0,45,90,180,180+45,180+90]: # [45,90,180,180+45,180+90] with teta = 180+90 only we have 596 output_plot
            if len(set(nodes) - set(list_fixed_nodes)) == len(set(nodes)): # i do not want to put load on a fixed edge
                count+=1
                top.setf(values=[1,1], nodes=nodes, tetas=[teta, teta])
                # top.fixed(list_nodes=list_fixed_nodes)
                top.optimize(store=True)
                top.plot()

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
				list_design_references.append(datetime_info+str(plot_reference+1))

				plot_reference+=1


    # 2 loads with opposite orientations
    for nodes in random.sample(opposite_side_nodes,2):
        for teta in [0,45,90,180,180+45,180+90]: # [45,90,180,180+45,180+90] with teta = 180+90 only we have 596 output_plot
            if len(set(nodes) - set(list_fixed_nodes)) == len(set(nodes)): # i do not want to put load on a fixed edge
                count+=1
                top.setf(values=[1,1], nodes=nodes, tetas=[teta, teta+180])
                # top.fixed(list_nodes=list_fixed_nodes)
                top.optimize(store=True)
                top.plot()

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
				list_design_references.append(datetime_info+str(plot_reference+1))

				plot_reference+=1



