import os

from randmig_functions import *
from plot_functions import * 
from velocity_and_acf_functions import * 
from event_verifi_plot_functions import *

#Total sim time
T_tot = 300 #[min]
#step size
dt = .05 #[min]

param_list = np.linspace(0.05, 5, 10)

for p in param_list:
    #set parameter values
    c = 500
    nu_pr = 80
    nu_cb = 100
    k = 5
    noise = 1
    kon = p
    koff = p
    params = [c, nu_pr, nu_cb, k, noise, kon, koff]

    save_path = '/Users/elizabethdavis/Desktop/Models/1D_CellSpringModel/figures/randmig_manycells/koffkon{}'.format(p)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #create new file to write parameters to
    param_vals = open(save_path+'/params.txt','w')
    file_lines = ['c: {}\n'.format(c), 'nu_pr: {}\n'.format(nu_pr), 'nu_cb: {}\n'.format(nu_cb), 'k: {}\n'.format(k), 'noise: {}\n'.format(noise), 'kon: {}\n'.format(kon), 'koff: {}\n'.format(koff)]
    #write lines to text file 
    param_vals.writelines(file_lines)
    param_vals.close() 

    num_walkers = 100
    #Where data is stored from all walkers in sim
    data_sim = []
    for walker in range(num_walkers):

        start_length = 100
        rand_start_ang = np.random.uniform(-np.pi, np.pi)

        xpr_0 = start_length * np.cos(rand_start_ang)
        ypr_0 = start_length * np.sin(rand_start_ang)
        xcb_0 = 0
        ycb_0 = 0

        force_0 = 1

        #set initial conditions
        y0 = [xpr_0, ypr_0, xcb_0, ycb_0]

        T, Y, force_each_step, fon_events, foff_events = EulerSolver(cell_motion, 0, T_tot, dt, y0, force_0, params)

        #Make dataframe of shape and motion metrics for track
        onewalker_df = make_motion_df(Y, dt)

        data_sim.append(onewalker_df)

    #plot cell body location of cells over time
    plot_cb_manycells(data_sim, save_path)

    #plot cell length over time
    plot_celllength_overtime(data_sim,T_tot,save_path)

    #Plot velocity acf
    plot_vel_acf_manycells(data_sim, save_path)

    #Plot boxplots of motion metrics
    make_motion_boxplots(data_sim, save_path)