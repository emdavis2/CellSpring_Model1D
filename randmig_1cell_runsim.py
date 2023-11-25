import os

from randmig_functions import *
from plot_functions import * 
from velocity_and_acf_functions import * 
from event_verifi_plot_functions import *

#Total sim time
T_tot = 9000 #[min]
#step size
dt = .05 #[min]

#Initialize starting positions of protrusion and cell body
xpr_0 = 100
ypr_0 = 100
xcb_0 = 0
ycb_0 = 0

force_0 = 0 #start force on

#set initial conditions
y0 = [xpr_0, ypr_0, xcb_0, ycb_0]

#set parameter values
c = 500
nu_pr = 80
nu_cb = 100
k = 5
noise = 1
kon = 0.3
koff = 0.3
params = [c, nu_pr, nu_cb, k, noise, kon, koff]

T, Y, force_each_step, fon_events, foff_events, fon_times, foff_times = EulerSolver(cell_motion, 0, T_tot, dt, y0, force_0, params)

save_path = '/Users/elizabethdavis/Desktop/Models/1D_CellSpringModel/figures/randmig_1cell'

if not os.path.exists(save_path):
  os.mkdir(save_path)

#create new file to write parameters to
param_vals = open(save_path+'/params.txt','w')
file_lines = ['c: {}\n'.format(c), 'nu_pr: {}\n'.format(nu_pr), 'nu_cb: {}\n'.format(nu_cb), 'k: {}\n'.format(k), 'noise: {}\n'.format(noise), 'kon: {}\n'.format(kon), 'koff: {}\n'.format(koff)]
#write lines to text file 
param_vals.writelines(file_lines)
param_vals.close() 

plot_cbpr(Y, save_path)

#Make movie of cell progressing over time
#make_movie(Y, T_tot, dt, -100, 100, save_path)

#Make dataframe of shape and motion metrics for track
onewalker_df = make_motion_df(Y, dt)

#Plot velocity acf
plot_vel_acf_onecell(onewalker_df['vx'],onewalker_df['vy'],save_path)

#plot cell length over time
#plot_celllength_overtime(onewalker_df,T_tot,save_path)

plot_timebtw_force_onoff(fon_events, foff_events, dt, kon, koff, save_path)
plot_cumnumevents(fon_events, foff_events, T_tot, save_path)
plot_events_5min_win(fon_events, foff_events, T_tot, dt, save_path)

plot_timebtw_force_onoff_method2(fon_times, foff_times, save_path)

# print('fon times are: ', fon_times)
# print('foff times are: ', foff_times)