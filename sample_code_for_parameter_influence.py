import os
import numpy as np
import matplotlib.pyplot as plt
import time
import elfi
import GPy
import operator
import subprocess
import shutil
import uuid
from scipy.spatial import distance

def get_model(bounds, y_obs):

    m = elfi.ElfiModel(name='poro_sim')
    
    #m = elfi.ElfiModel(name='poro_sim')
    # set distribution lower bound and scale based on bounds
    elfi.Prior('uniform', bounds['sigma_H'][0], bounds['sigma_H'][1]-bounds['sigma_H'][0], model=m, name='sigma_H')
    # update sigma_h and sigma_v upper bounds based on sigma_H
    elfi.Operation(np.minimum, bounds['sigma_v'][1], m['sigma_H'], name='upper_v')
    elfi.Operation(np.minimum, bounds['sigma_h'][1], m['sigma_H'], name='upper_h')
    # calculate scales
    elfi.Operation(operator.sub, m['upper_v'], bounds['sigma_v'][0], name='scale_v')
    elfi.Operation(operator.sub, m['upper_h'], bounds['sigma_h'][0], name='scale_h')
    elfi.Prior('uniform', bounds['sigma_v'][0], m['scale_v'], model=m, name='sigma_v')
    elfi.Prior('uniform', bounds['sigma_h'][0], m['scale_h'], model=m, name='sigma_h')
    # calculate overpressure lower bound based on sigma parameters
    elfi.Operation(operator.sub, m['sigma_H'], m['sigma_h'], name='diff')  # calculate difference between sigma values

    elfi.Operation(operator.mul, m['diff'], 0.5, name='bound')  # bound = 0.5 * difference
    # calculate overpressure lower bound and scale based on prior bounds and above
    elfi.Operation(np.maximum, bounds['overpressure'][0], m['bound'], name='lower')  # overpressure lower bound
    elfi.Operation(operator.sub, bounds['overpressure'][1], m['lower'], name='scale')  # calculate scale based on upper and lower bound
    elfi.Prior('uniform', m['lower'], m['scale'], model=m, name='overpressure')


    # set distribution lower bound and scale based on bounds
    elfi.Prior('uniform', bounds['a1'][0], bounds['a1'][1]-bounds['a1'][0], model=m, name='a1')
    elfi.Prior('uniform', bounds['a2'][0], bounds['a2'][1]-bounds['a2'][0], model=m, name='a2')
    elfi.Prior('uniform', bounds['a3'][0], bounds['a3'][1]-bounds['a3'][0], model=m, name='a3')

    # simulator
    sim_call_poro = './poro --nodes=1 --ntasks=8 > poro_dump.txt'
    #sim_call_poro = 'srun ./poro > poro_dump.txt'
    poro_sim = elfi.tools.external_operation(sim_call_poro, stdout = False, prepare_inputs = prepare_inputs, process_result = process_result);
    # vectorization of the external command
    poro_sim_vector = elfi.tools.vectorize(poro_sim);
    #poro_node = elfi.Simulator(poro_sim_vector, m['a1'], m['a2'], m['a3'], name='poro', observed = y_obs);

    poro_node = elfi.Simulator(poro_sim_vector, m['sigma_H'], m['sigma_v'], m['sigma_h'], m['overpressure'], m['a1'], m['a2'], m['a3'], name='poro', observed = y_obs);

    # distance
    elfi.Distance('euclidean', m['poro'], name='distance_1')
    elfi.Operation(np.log, m['distance_1'], name='log_d')

    return m


def prepare_inputs(*inputs, **kwinputs):
    # Store the current working directory
    cwd = os.getcwd()
    kwinputs['cwd'] = cwd

    # Print current working directory and inputs
    print('Current directory : ', cwd)
    print('printing inputs', inputs)

    # Convert the inputs tuple to a list
    input_list = list(inputs)

    input_filename2 = 'stress_file.dat'
    
    

    # Ensure at least two inputs are provided
    if len(input_list) < 7:
        raise ValueError("At least three input values are required")

    # Define the input file name as 'material.dat'
    input_filename = 'material.dat'

    # Generate a unique run directory name
    rundir = 'MA2_rundir_{}'.format(uuid.uuid4())

    # Define the copy command to include required files
    copy_command_1 = 'cp poro general.dat grid_size.dat stress_file.dat boundary.dat elements.dat coordinates.dat connections.dat cat2ev_tensor.f cat2ev_tensor ia.in ja.in' + ' ' + rundir

    # Create the run directory
    os.mkdir(rundir)

    # Execute the copy command to copy files into the run directory
    os.system(copy_command_1)

    # Change to the new run directory
    os.chdir(rundir)
    kwinputs['rundir'] = rundir

    # Print the new current directory and input filename
    print('Current directory : ', os.getcwd())
    print('printing input filename ', input_filename)


    print('Current directory : ', os.getcwd())
    print('printing input filename ',input_filename2)
    np.savetxt(input_filename2, input_list[:4], fmt='%d')




    # Define constant values to be added to the second line
    Dens = "2.7e+3"
    Young = "40.0e+9"
    Pois = "0.2"
    M_Biot = "15.0e+9"
    ksi_0 = "-0.95"
    Cd = "3.0e+3"
    C1 = "1.0e-5"
    C2 = "0.03"
    tau_d = "100000.0"
    alpha = "0.08"
    Cv = "1.0e-11"
    amp = "0.1"

    al_Biot = "1.0"

    a_1 = round(input_list[4], 2)
    a_2 = round(input_list[5], 0)
    a_3 = round(input_list[6], 2)
    initial_permeability = a_1 * 10 ** -a_2
    print("a 1 : ", a_1)
    permeability_growth_factor = str(a_3)


    # Define the content to be written
    first_line = "2\n"
    second_line = (
        "{}  {}  {}  {}   {}   {} {}  {}    "
        "{}    {}    {}   {}   {}  {}  {}    ! granite".format(Dens, Young, Pois, M_Biot, ksi_0, Cd, C1, C2, tau_d, alpha, amp, Cv, initial_permeability, permeability_growth_factor, al_Biot)
    )

    # File name
    file_name = 'material.dat'

    # Write the content to the file
    with open(input_filename, 'w') as file:
        file.write(first_line)
        file.write(second_line + "\n")
        file.write(second_line + "\n")
    return inputs, kwinputs








def process_result(completed_process, *inputs, **kwinputs):

    print('processing outputs........')
    
    print('Current directory : ', os.getcwd())

    os.system('gfortran -o cat2ev_tensor cat2ev_tensor.f')
    os.system('./cat2ev_tensor')
    command_compile = 'gfortran -o cat2ev_tensor cat2ev_tensor.f'
    command = './cat2ev_tensor'

    os.system('gfortran -o cat2ev_tensor cat2ev_tensor.f')
    os.system('./cat2ev_tensor')
    command_compile = 'gfortran -o cat2ev_tensor cat2ev_tensor.f'
    command = './cat2ev_tensor'


    subprocess.run(command_compile, shell=True)
    subprocess.run(command, shell=True)

    rundir = kwinputs['rundir']
    simulator_path = kwinputs['cwd']
    
    current_path_new = os.path.join(simulator_path, rundir)
    #print('Current directory : ', current_path_new)

    output_filename = 'events'
    file_path = os.path.join(current_path_new, output_filename)
    print('file_path : ', file_path)

    if not os.path.exists(file_path):
        print('events file was not created')
        data = np.array([0,0,0,0,0,0,0,0,0])
    else:
        data = process_events(file_path)

    os.chdir(simulator_path)
    return data


def process_events(file_path):
    
    #print('Current directory : ', os.getcwd())
    data = np.loadtxt(file_path)
    
    if data.ndim == 1:
        if os.path.getsize(file_path) == 0:
            sim_tp_data_final =np.array([0,0,0,0,0,0,0,0,0])
        else:
            sim_tp_data_final = np.array([1,1,1,1,1,1,1,1,1])

    else:
        q1_num=0
        q2_num=0
        q3_num=0
        q4_num=0
        q5_num=0
        q6_num=0
        q7_num=0
        q8_num=0

        q1_TP=[0]
        q2_TP=[0]
        q3_TP=[0]
        q4_TP=[0]
        q5_TP=[0]
        q6_TP=[0]
        q7_TP=[0]
        q8_TP=[0]

        TP = [0]

        X = data[:, [0,1,2,3,4,5]]

        x = X[:,1]
        y = X[:,2]
        z = X[:,3]
        tp = X[:,4]

        for i in range(X.shape[0]):

            if x[i] >=1000 and y[i] >=1000 and z[i] >=1000:
                q1_num+=1
                q1_TP.append(tp[i])

            elif x[i] >=1000 and y[i] <=1000 and z[i] >=1000:
                q2_num+=1
                q2_TP.append(tp[i])

            elif x[i] >=1000 and y[i] >=1000 and z[i] <=1000:
                q3_num+=1
                q3_TP.append(tp[i])

            elif x[i] >=1000 and y[i] <=1000 and z[i] <=1000:
                q4_num+=1
                q4_TP.append(tp[i])

            elif x[i] <=1000 and y[i] >=1000 and z[i] <=1000:
                q5_num+=1
                q5_TP.append(tp[i])

            elif x[i] <=1000 and y[i] <=1000 and z[i] <=1000:
                q6_num+=1
                q6_TP.append(tp[i])

            elif x[i] <=1000 and y[i] >=1000 and z[i] >=1000:
                q7_num+=1
                q7_TP.append(tp[i])

            elif x[i] <=1000 and y[i] <=1000 and z[i] >=1000:
                q8_num+=1
                q8_TP.append(tp[i])

            else:
                pass

        q_num_sum = q1_num + q2_num + q3_num + q4_num + q5_num + q6_num + q7_num + q8_num

        sim_data_final = np.array([q_num_sum, q1_num, q2_num, q3_num, q4_num, q5_num, q6_num, q7_num, q8_num])

        q1_tp = np.mean(np.array([q1_TP]))
        q2_tp = np.mean(np.array([q2_TP]))
        q3_tp = np.mean(np.array([q3_TP]))
        q4_tp = np.mean(np.array([q4_TP]))
        q5_tp = np.mean(np.array([q5_TP]))
        q6_tp = np.mean(np.array([q6_TP]))
        q7_tp = np.mean(np.array([q7_TP]))
        q8_tp = np.mean(np.array([q8_TP]))


        q_tp_sum = q1_tp + q2_tp + q3_tp + q4_tp + q5_tp + q6_tp + q7_tp + q8_tp

        sim_tp_data_final = np.array([q_tp_sum, q1_tp, q2_tp, q3_tp, q4_tp, q5_tp, q6_tp, q7_tp, q8_tp])

    return sim_tp_data_final 


def create_target_model(m, bounds):
    input_dim = len(m.parameter_names)
    # create the SE-kernel
    kernel_seard = GPy.kern.RBF(input_dim = input_dim, ARD = True)
    kernel_seard.lengthscale = 5
    kernel_seard.lengthscale.set_prior(GPy.priors.Gamma(2,0.04), warning = False)
    kernel_seard = kernel_seard + GPy.kern.Bias(input_dim = input_dim)
    target_model = elfi.GPyRegression(m.parameter_names, bounds=bounds, kernel=kernel_seard)
    return target_model


def main():

    seed = 583457
    filename_model = 'data_save_final'
    filename_model_other = 'posterior_data_save_final'
    num_parallel_batches = 1  
    num_init_evidence = 64
    num_of_evidence = 128
    num_samples = 1000  # num posterior samples
    num_warmup = 500

    # observed data
    y_obs = np.array([8418.79, 840.66, 895.12, 906.76, 967.13, 1438.13, 989.14, 1329.07, 1052.78])
    
    # parameter bounds
    bounds = {}

    bounds['sigma_H'] = (200, 340)
    bounds['sigma_v'] = (140, 240)
    bounds['sigma_h'] = (60, 160)
    bounds['overpressure'] = (20, 120)

    bounds['a1'] = (1, 10)
    bounds['a2'] = (13, 20)
    bounds['a3'] = (1, 10)

    m = get_model(bounds, y_obs)
    print(m.parameter_names)
  
    for i in range(140, 241, 10):

        gen_num = 3
        data_true = m.generate(gen_num, with_values={'sigma_H':340, 'sigma_v':i, 'sigma_h':140, 'overpressure':140,'a1':3, 'a2': 17, 'a3': 16})


if __name__ == '__main__':
    elfi.set_client('multiprocessing')
    main()
