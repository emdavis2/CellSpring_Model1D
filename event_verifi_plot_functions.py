import numpy as np
import matplotlib.pyplot as plt

#function that plots histogram of time between events for both force on and force off
#Inputs:
# fon_events => number of force on events at each time step (type: list of ints with len=t_end/dt)
# foff_events => number of force off events at each time step (type: list of ints with len=t_end/dt)
# dt => step size in solver (type: float)
# save_path => path to folder where plot will be saved to (type: string)
#Outputs:
# figures of the histogram of time between both fon and foff events
def plot_timebtw_force_onoff(fon_events, foff_events, dt, save_path):
    time_between_fon_events = np.diff(np.nonzero(fon_events)[0])*dt
    plt.hist(time_between_fon_events,density=False)
    plt.title('time between force on events')
    plt.xlabel('time (min)')
    plt.savefig(save_path+'/time_between_fon.png')
    plt.clf()

    time_between_foff_events = np.diff(np.nonzero(foff_events)[0])*dt
    plt.hist(time_between_foff_events,density=False)
    plt.title('Time between force off events')
    plt.xlabel('time (min)')
    plt.savefig(save_path+'/time_between_foff.png')
    plt.clf()

###################################################################################################################################################

#function that plots the cumulative number of force on and off events over time
#Inputs:
# fon_events => number of force on events at each time step (type: list of ints with len=t_end/dt)
# foff_events => number of force off events at each time step (type: list of ints with len=t_end/dt)
# t_end => end time point to run simulation (type: int)
# save_path => path to folder where plot will be saved to (type: string)
#Outputs:
# figures of the cumulative sum of number of evetns over the length of the simulation
def plot_cumnumevents(fon_events, foff_events, t_end, save_path):
    plt.plot(np.linspace(0,t_end,len(fon_events)),np.cumsum(fon_events))
    plt.xlabel('time (min)')
    plt.ylabel('Number of events for force on')
    plt.savefig(save_path+'/cum_num_fon.png')
    plt.clf()

    plt.plot(np.linspace(0,t_end,len(fon_events)),np.cumsum(foff_events))
    plt.xlabel('time (min)')
    plt.ylabel('Number of events for force off')
    plt.savefig(save_path+'/cum_num_foff.png')
    plt.clf()
    
###################################################################################################################################################

#function to plot histograms of the number of events that occur in a 5 minute window
#Inputs:
# fon_events => number of force on events at each time step (type: list of ints with len=t_end/dt)
# foff_events => number of force off events at each time step (type: list of ints with len=t_end/dt)
# t_end => end time point to run simulation (type: int)
# dt => step size in solver (type: float)
# save_path => path to folder where plot will be saved to (type: string)
#Outputs:
# figures of histograms for the number of force on and force off events that occur in a 5 minute window over the simulation
def plot_events_5min_win(fon_events, foff_events, t_end, dt, save_path):
    fon_events_min = np.reshape(fon_events, (int(t_end/5),int(5/dt))) #reshape to find sum over window of events per minute
    fon_events_min = np.sum(fon_events_min, axis=1) #find sum over window
    plt.hist(fon_events_min, density=True)
    plt.title('Number of force on events in 5 minute window')
    plt.savefig(save_path+'/num_fon_5minwin.png')
    plt.clf()

    foff_events_min = np.reshape(foff_events, (int(t_end/5),int(5/dt))) #reshape to find sum over window of events per minute
    foff_events_min = np.sum(foff_events_min, axis=1) #find sum over window
    plt.hist(foff_events_min, density=True)
    plt.title('Number of force off events in 5 minute window')
    plt.savefig(save_path+'/num_foff_5minwin.png')
    plt.clf()