import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import seaborn as sns

from velocity_and_acf_functions import *

#function that plots centroid position of the cell over time
#Inputs:
# Y => x and y positions for protrusion and cell body for each time point with (type: ndarray with shape=(len(T),4))
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# plot of cell body and protrusion position saved as png to directory specified in save_path variable
def plot_cbpr(Y,save_path):
    Xcb = Y[:,2]
    Ycb = Y[:,3]
    Xpr = Y[:,0]
    Ypr = Y[:,1]

    plt.plot(Xcb, Ycb)
    plt.plot(Xpr, Ypr)
    plt.xlabel(r'x position [$\mu m$]')
    plt.ylabel(r'y position [$\mu m$]')
    plt.title('cell body and protrusion position of cell over time')
    plt.savefig(save_path+'/cb_pos.png')
    plt.clf()

###################################################################################################################################################

#function that plots centroid position of many cells over time
#Inputs:
# data_sim => list of dataframes that contains x, y coordinates of cell body as well as motion metrics for each track (type: list of pandas dataframes)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# plot of cell body position saved as png to directory specified in save_path variable
def plot_cb_manycells(data_sim,save_path):
    for df in data_sim:
        plt.plot(df['Xcb'],df['Ycb']) #these positins should already be smoothed
    plt.xlabel(r'x position [$\mu m$]')
    plt.ylabel(r'y position [$\mu m$]')
    plt.title('cell body position of cell over time')
    plt.savefig(save_path+'/cb_pos_manycells.png')
    plt.clf()

###################################################################################################################################################

#function that plots centroid position of many cells over time
#Inputs:
# data_sim => list of dataframes that contains x, y coordinates of cell body as well as motion metrics for each track (type: list of pandas dataframes)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# plot of cell body position saved as png to directory specified in save_path variable
def plot_celllength_overtime(data_sim,t_end,save_path):
    for df in data_sim:
        cell_length = np.sqrt((df['Xpr']-df['Xcb'])**2 + (df['Ypr']-df['Ycb'])**2)
        plt.plot(np.linspace(0,t_end,len(cell_length)),cell_length) 
    plt.xlabel(r'x position [$\mu m$]')
    plt.ylabel(r'y position [$\mu m$]')
    plt.title('length of cell over time')
    plt.savefig(save_path+'/cell_length_manycells.png')
    plt.clf()

###################################################################################################################################################

#function that makes a mp4 video of the cell evolving over time - note: movie shape is always square...doesn't have to be but is for now
#Inputs:
# Y => x and y positions for protrusion and cell body for each time point with (type: ndarray with shape=(len(T),4))
# t_end => end time point to run simulation (type: int)
# dt => step size in solver (type: float)
# xy_min => lower bound for x and y axis limits (type: float)
# xy_max => upper bound for x and y axis limits (type: float)
# save_path => path to folder where plot will be saved to (type: string)
#Output:
# movie of cell evolving over time saved as mp4
def make_movie(Y,t_end,dt,xy_min,xy_max,save_path):
    Xcb = Y[:,2]
    Ycb = Y[:,3]
    Xpr = Y[:,0]
    Ypr = Y[:,1]

    Cell_dict_x = {}
    Cell_dict_y = {}

    for i in range(len(Xpr)):
        Cell_dict_x[i] = [Xpr[i], Xcb[i]]
        Cell_dict_y[i] = [Ypr[i], Ycb[i]]


    fig = plt.figure()
    ax = plt.axes(xlim=(xy_min, xy_max), ylim=(xy_min, xy_max))
    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x_ti = Cell_dict_x[i]
        y_ti = Cell_dict_y[i]

        line.set_data(x_ti, y_ti)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=int((t_end/dt)), interval=20, blit=True)

    FFwriter = FFMpegWriter(fps=20)
    anim.save(save_path + '/1DCellSpring.mp4', writer=FFwriter)

###################################################################################################################################################

#function to plot velocity acf for one cell
#Inputs:
# x_vel => x component of velocity (type: pandas series of floats)
# y_vel => y componenet of velocity (type: pandas series of floats)
# save_path => path to folder where plot will be saved to (type: string)
# Optional:
# min_track_length => specified track length to cut off autocorrelation results (what value of tau to cut off at)
#Outputs:
# figure of velocity autocorrelation for lags up to min_track_length-4
def plot_vel_acf_onecell(x_vel, y_vel, save_path, min_track_length=30):
    combined = make_comb_df(x_vel, y_vel)
    poslagsmean, Nposlags, neglagsmean, Nneglags = xcorr_vector(combined, min_track_length)

    plt.plot(poslagsmean,label = "positive lag")
    plt.hlines(y=0,xmin=0,xmax=100,color='k')
    plt.xlim(0,min_track_length-4)
    plt.ylim(-0.5,1)
    plt.xlabel('lag (5 min)')
    plt.title("Autocorrelaton velocity")
    plt.savefig(save_path+'/velocity_acf.png')
    plt.clf()

###################################################################################################################################################

#function to plot average velocity acf across lags from many cells
#Inputs:
# data_sim => list of dataframes that contains x, y coordinates of cell body as well as motion metrics for each track (type: list of pandas dataframes)
# save_path => path to folder where plot will be saved to (type: string)
# Optional:
# min_track_length => specified track length to cut off autocorrelation results (what value of tau to cut off at)
#Outputs:
# figure of average velocity autocorrelation across lags from many cells up to min_track_length-4
def plot_vel_acf_manycells(data_sim, save_path, min_track_length=30):
    poslagaverage = np.zeros(30000)
    Nposlagtotal = np.zeros(30000)
    all_ac = []
    for df in data_sim:
        combined = make_comb_df(df['vx'], df['vy'])
        poslagsmean, Nposlags, neglagsmean, Nneglags = xcorr_vector(combined, min_track_length)

        #remove nans here
        poslagsmean[np.isnan(poslagsmean)] = 0
        all_ac.append(poslagsmean)
        poslagaverage[0:len(poslagsmean)] += poslagsmean # Nposlags*poslagsmean

    poslagaverage /= len(data_sim) #Nposlagtotal

    std_err = np.std(all_ac,axis=0,ddof=1)/np.sqrt(np.shape(all_ac)[0])

    plt.errorbar(np.arange(0,min_track_length-4),poslagaverage[0:min_track_length-4],yerr=std_err)
    plt.hlines(y=0,xmin=0,xmax=100,color='k')
    plt.xlim(0,min_track_length-4)
    plt.ylim(-0.5,1)
    plt.xlabel('lag (5 min)')
    plt.title("Autocorrelaton velocity")
    plt.savefig(save_path+'/velocity_acf_avgcells.png')
    plt.clf()

###################################################################################################################################################

#function to plot boxplots for motion metrics calculated in make_motion_df()
#Inputs:
# data_sim => list of dataframes that contains x, y coordinates of cell centroid as well shape and motion metrics for each track (type: list of pandas dataframes)
# save_path => path to folder where plot will be saved to (type: string)
#Outputs:
# figures of boxplots for D/T, Speed, Area, and solidity
def make_motion_boxplots(data_sim, save_path):
    DT = []
    speed = []
    for df in data_sim:
        DT.append(df['DoverT'][0])
        speed.append(df['Speed'][0])

    sns.boxplot(data=DT)
    plt.xlabel('From model with {} cells simulated'.format(len(data_sim)))
    plt.ylabel('D/T')
    plt.savefig(save_path+'/DT_boxplot.png')
    plt.clf()

    sns.boxplot(data=np.array(speed)/5) #because sampling rate is every 5 minutes
    plt.xlabel('From model with {} cells simulated'.format(len(data_sim)))
    plt.ylabel('Speed $\mu m$/min')
    plt.savefig(save_path+'/Speed_boxplot.png')
    plt.clf()