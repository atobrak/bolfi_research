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


def get_model(bounds, y_obs):

    m = elfi.ElfiModel(name='poro')
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
    elfi.Operation(np.maximum, bounds['over_pressure'][0], m['bound'], name='lower')  # overpressure lower bound
    elfi.Operation(operator.sub, bounds['over_pressure'][1], m['lower'], name='scale')  # calculate scale based on upper and lower bound
    elfi.Prior('uniform', m['lower'], m['scale'], model=m, name='over_pressure')

    # simulator
    sim_call_poro = './poro --nodes=1 --ntasks=8 > poro_dump.txt'
    #sim_call_poro = 'srun ./poro > poro_dump.txt'
    poro_sim = elfi.tools.external_operation(sim_call_poro, stdout = False, prepare_inputs = prepare_inputs, process_result = process_result);
    # vectorization of the external command
    poro_sim_vector = elfi.tools.vectorize(poro_sim);
    poro_node = elfi.Simulator(poro_sim_vector, m['sigma_H'], m['sigma_v'], m['sigma_h'], m['over_pressure'], name='poro', observed = y_obs);

    # distance
    elfi.Distance('euclidean', m['poro'], name='distance_1')
    elfi.Operation(np.log, m['distance_1'], name='log_d')

    return m


def prepare_inputs(*inputs, **kwinputs):

    cwd = os.getcwd()
    kwinputs['cwd'] = cwd
    
    print('Current directory : ', cwd)
    print('printing inputs',inputs)
    input_list = list(inputs)
    
    input_filename = 'stress_file.dat'
    rundir = 'MA2_rundir_{}'.format(uuid.uuid4())

    copy_command_1 = 'cp poro general.dat grid_size.dat material.dat boundary.dat elements.dat coordinates.dat connections.dat cat2ev_tensor.f cat2ev_tensor ia.in ja.in' + ' ' + rundir

    os.mkdir(rundir)
    os.system(copy_command_1)
    os.chdir(rundir)
    kwinputs['rundir'] = rundir

    print('Current directory : ', os.getcwd())
    print('printing input filename ',input_filename)
    np.savetxt(input_filename, input_list[:4], fmt='%d')
    return inputs, kwinputs


def process_result(completed_process, *inputs, **kwinputs):

    print('processing outputs........')
    
    print('Current directory : ', os.getcwd())
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
    num_init_evidence = 40
    num_of_evidence = 200
    num_samples = 1000  # num posterior samples
    num_warmup = 500

    # observed data
    y_obs = np.array([8418.79, 840.66, 895.12, 906.76, 967.13, 1438.13, 989.14, 1329.07, 1052.78])

    # parameter bounds
    bounds = {}
    bounds['sigma_H'] = (200, 340)
    bounds['sigma_v'] = (140, 240)
    bounds['sigma_h'] = (60, 140)
    bounds['over_pressure'] = (70, 140)

    m = get_model(bounds, y_obs)
    print(m.parameter_names)
    
    #gen_num = 2
    #data_rand = m.generate(gen_num)
    #data_true = m.generate(gen_num, with_values={'sigma_H':250, 'sigma_v': 160, 'sigma_h': 100, 'over_pressure': 105})

    #print(data_rand['poro'])
    #np.save('data_rand.npy', data_rand)

    #print(data_true['poro'])
    #np.save('data_true.npy', data_true)
    
    #precomputed = np.load('data_rand_new4.npy', allow_pickle=True)
    #precomputed = precomputed.item()
    #data_pre = {'sigma_H': precomputed['sigma_H'], 'sigma_v': precomputed['sigma_v'],'sigma_h':precomputed['sigma_h'], 'over_pressure':precomputed['over_pressure'], 'log_d': np.log(precomputed['distance_1'])}


    
    # BOLFI
    target_model = create_target_model(m, bounds)
    prior = elfi.model.extensions.ModelPrior(m)
    acq = elfi.methods.bo.acquisition.RandMaxVar(model=target_model, prior=prior, sampler='metropolis', n_samples=num_samples, warmup=num_warmup, init_from_prior=True)

    bolfi = elfi.BOLFI(m['log_d'],
        batch_size=1,
        max_parallel_batches=num_parallel_batches,
        initial_evidence=num_init_evidence, #data_pre,
        bounds=bounds,
        target_model=target_model,
        acquisition_method=acq,
        seed=seed)

    # run BOLFI
    start_time = time.time()
    post = bolfi.fit(n_evidence=num_of_evidence)
    end_time = time.time()
    execution_time = end_time - start_time
    print('BOLFI execution time [seconds] : ', execution_time)

    parameter_names_array = m.parameter_names    
    print("parameter names : ", m.parameter_names)
    print("parameter names array : ", parameter_names_array)

    print("test_bound : ",bounds['over_pressure'])
    xlim = []

    for i in range(len(parameter_names_array)):
        xlim.append(list(bounds[parameter_names_array[i]]))


    xlim = np.array(xlim) 
    resolution = 50
    xp_0 = np.linspace(xlim[0,0], xlim[0,1], resolution)
    xp_1 = np.linspace(xlim[1,0], xlim[1,1], resolution)
    xp_2 = np.linspace(xlim[2,0], xlim[2,1], resolution)
    xp_3 = np.linspace(xlim[3,0], xlim[3,1], resolution)

    

    ###################################################################################################
    ###################################################################################################
    

    plot_model = bolfi.target_model

    X_evidence = plot_model.X
    Y_evidence = plot_model.Y

    Constant = X_evidence[np.argmin(Y_evidence), :]


    # sigma_H and sigma_v
    X1, X3 = np.meshgrid(xp_1, xp_3)

    Predictors_1 = np.tile(Constant, (resolution * resolution, 1))
    print(Predictors_1.shape)
    Predictors_1[:, 1] = X1.ravel()
    Predictors_1[:, 3] = X3.ravel()

    print(X1.ravel().shape)
    print(X3.ravel().shape)

    print(Predictors_1.shape)

    yp_1 = np.array(plot_model.predict(np.array(Predictors_1)))

    print(yp_1[0])
    print(yp_1[0].shape)

    z_1 = np.array(plot_model.predict(np.array(Predictors_1)))[0].reshape(resolution, resolution)

    np.save("X1_1.npy", X1)
    np.save("X3_1.npy", X3)
    np.save("z_1.npy", z_1)


    plt.contourf(X1, X3, z_1, cmap="Blues")
    plt.savefig("new_test_c_plot_1.png")
    plt.close()

    # overpressure and sigma_h
    X0, X2 = np.meshgrid(xp_0, xp_2)
    Predictors_2 = np.tile(Constant, (resolution * resolution, 1))
    Predictors_2[:, 0] = X0.ravel()
    Predictors_2[:, 2] = X2.ravel()

    z_2 = np.array(plot_model.predict(np.array(Predictors_2)))[0].reshape(resolution, resolution)

    np.save("X0_2.npy", X0)
    np.save("X2_2.npy", X2)
    np.save("z_2.npy", z_2)


    plt.contourf(X0, X2, z_2, cmap="Blues")
    plt.savefig("new_test_c_plot_2.png")
    plt.close()

    # overpressure and sigma_v
    X0, X3 = np.meshgrid(xp_0, xp_3)
    Predictors_3 = np.tile(Constant, (resolution * resolution, 1))
    Predictors_3[:, 0] = X0.ravel()
    Predictors_3[:, 3] = X3.ravel()

    z_3 = np.array(plot_model.predict(np.array(Predictors_3)))[0].reshape(resolution, resolution)

    np.save("X0_3.npy", X0)
    np.save("X3_3.npy", X3)
    np.save("z_3.npy", z_3)


    plt.contourf(X0, X3, z_3, cmap="Blues")
    plt.savefig("new_test_c_plot_3.png")
    plt.close()

    # overpressure and sigma_H
    X0, X1 = np.meshgrid(xp_0, xp_1)
    Predictors_4 = np.tile(Constant, (resolution * resolution, 1))
    Predictors_4[:, 0] = X0.ravel()
    Predictors_4[:, 1] = X1.ravel()

    z_4 = np.array(plot_model.predict(np.array(Predictors_4)))[0].reshape(resolution, resolution)

    np.save("X0_4.npy", X0)
    np.save("X1_4.npy", X1)
    np.save("z_4.npy", z_4)


    plt.contourf(X0, X1, z_4, cmap="Blues")
    plt.savefig("new_test_c_plot_4.png")
    plt.close()

    #sigma_H and sigma_h
    X1, X2 = np.meshgrid(xp_1, xp_2)
    Predictors_5 = np.tile(Constant, (resolution * resolution, 1))
    Predictors_5[:, 1] = X1.ravel()
    Predictors_5[:, 2] = X2.ravel()

    z_5 = np.array(plot_model.predict(np.array(Predictors_5)))[0].reshape(resolution, resolution)

    np.save("X1_5.npy", X1)
    np.save("X2_5.npy", X2)
    np.save("z_5.npy", z_5)


    plt.contourf(X1, X2, z_5, cmap="Blues")
    plt.savefig("new_test_c_plot_5.png")
    plt.close()

    #sigma_h and sigma_v
    X2, X3 = np.meshgrid(xp_2, xp_3)
    Predictors_6 = np.tile(Constant, (resolution * resolution, 1))
    Predictors_6[:, 2] = X2.ravel()
    Predictors_6[:, 3] = X3.ravel()

    z_6 = np.array(plot_model.predict(np.array(Predictors_6)))[0].reshape(resolution, resolution)
    
    np.save("X2_6.npy", X2)
    np.save("X3_6.npy", X3)
    np.save("z_6.npy", z_6)


    plt.contourf(X2, X3, z_6, cmap="Blues")
    plt.savefig("new_test_c_plot_6.png")
    plt.close()


    

    

    """ 
    cset = plt.contourf(xp_1, xp_3, yp_plot, levels=10, cmap='Blues')
    plt.scatter(plot_model.X[:, 1], plot_model.X[:, 3], s=10, zorder=10, facecolors="r", label="evidence")

    plt.xlim(xlim[1])
    plt.ylim(xlim[3])

    plt.xlabel("sigma_H")
    plt.ylabel("sigma_v")
    plt.colorbar(cset)
    #plt.legend(loc="center left")
    plt.savefig("test_new_contour_plot.png")
    plt.close()

    """

    ####################################################################################################################
    ####################################################################################################################
    

    
    

    """

    cset = plt.contourf(xp_0, xp_2, yp_plot_2, levels=10, cmap='Blues')
    plt.scatter(plot_model_2.X[:, 0], plot_model_2.X[:, 2], s=10, zorder=10, facecolors="r", label="evidence")

    plt.xlim(xlim[0])
    plt.ylim(xlim[2])

    plt.xlabel("over_pressure")
    plt.ylabel("sigma_v")
    plt.colorbar(cset)
    #plt.legend(loc="center left")
    plt.savefig("test_new_contour_plot_2.png")
    plt.close()

    """

    ####################################################################################################################
    ####################################################################################################################

    

    res1 = bolfi.extract_result()
    print(res1.x_min)
    
    # save model    
    dist_model = bolfi.target_model._gp
    np.savez(filename_model, X=dist_model.X, Y=dist_model.Y, params=dist_model.param_array)

    cont_plot = bolfi.plot_gp()
    plt.savefig('contour_plot_final.png')

    bolfi.plot_state();
    plt.savefig('bolfi_states_final.png')

    bolfi.plot_discrepancy();
    plt.savefig('bolfi_discrepancy_final.png')

    # posterior sample
    result_BOLFI = bolfi.sample(num_samples)
    print(result_BOLFI)
    np.save(filename_model_other, result_BOLFI.samples_array)

    result_BOLFI.plot_traces();
    plt.savefig('bolfi_traces_final.png')
    
    result_BOLFI.plot_marginals();
    plt.savefig('bolfi_marginals_final.png')

    

    


if __name__ == '__main__':
    elfi.set_client('multiprocessing')
    main()
