import numpy as np
import math

#function that updates whether force at protursion is on or off based off rate for force on and off
#Inputs:
# force => previous state of force at previous time point, either 1 for on or 0 for off (type: int)
# dt => step size in solver (type: float)
#Optional:
# kon => force on rate (1/min) (type: float)
# koff => force off rate (1/min) (type: float)
#Outputs:
# force => updated force state, either 1 for on or 0 for off (type: int)
# fon_e => whether force was swithced from off to on at this time step (0 is no and 1 is yes) (type: int)
# foff_e => whether force was swithced from on to off at this time step (0 is no and 1 is yes) (type: int)
def update_F(force, dt, time, kon=0.3, koff=0.3):
  if force == 0: #force is off
    prob_on = kon*dt
    rand_num = np.random.uniform()
    if rand_num < prob_on:
      force = 1
      fon_e = 1
      foff_e = 0
      fon_t = time
      foff_t = 0
    else:
      fon_e = 0
      foff_e = 0
      fon_t = 0
      foff_t = 0
  elif force == 1:
    prob_off = koff*dt
    rand_num = np.random.uniform()
    if rand_num < prob_off:
      force = 0
      foff_e = 1
      fon_e = 0
      foff_t = time
      fon_t = 0
    else:
      foff_e = 0
      fon_e = 0
      foff_t = 0
      fon_t = 0

  return force, fon_e, foff_e, fon_t, foff_t

###################################################################################################################################################

#function that outputs the right hand side of the ODE for the equations of motion for protrusion and cell body - this is what is being solved with the Euler solver
#Inputs:
# dt => step size in solver (type: float)
# y_val => x and y positions of the protrusion and cell body together in one list (type: list of len=4)
# force_prev => state of force at previous time step - 1 for on and 0 for off (type: int)
# params => list of params for simulation (type: list of floats) params = [c, nu_pr, nu_cb, k, noise, kon, koff]
#Outputs:
# dr => the right hand side of the ODE for the equations of motion for protrusion and cell body where dx and dy are together in one array (type: array with len=4)
# force => updated state of force - 1 for on and 0 for off (type: int)
# fon_e => whether force was swithced from off to on at this time step (0 is no and 1 is yes) (type: int)
# foff_e => whether force was swithced from on to off at this time step (0 is no and 1 is yes) (type: int)
def cell_motion(dt,y_val,force_prev, params,time):
  xpr = y_val[0]
  ypr = y_val[1]
  xcb = y_val[2]
  ycb = y_val[3]

  # #Define parameters
  # nu_pr = 80 #viscous friction factor [nN min um^(-1)]
  # nu_cb = 100
  # k = 5 #spring constant of spring connecting cell body and protrusion
  # noise = 1 # scale parameter for noise from normal distribution
  # c = 500 #force constant

  c = params[0]
  nu_pr = params[1]
  nu_cb = params[2]
  k = params[3]
  noise = params[4]
  kon = params[5]
  koff = params[6]

  #theta_pr = math.atan2(ypr,xpr)
  #theta_cb = math.atan2(ycb,xcb)

  R = np.sqrt((xpr-xcb)**2 + (ypr-ycb)**2) #distance between cell body and protrusion

  F_spring = k*R

  xR = xpr - xcb
  yR = ypr - ycb

  force, fon_e, foff_e, fon_t, foff_t = update_F(force_prev, dt, time, kon, koff)

  thetaR = math.atan2(yR,xR)

  F_springx = F_spring * np.cos(thetaR)
  F_springy = F_spring * np.sin(thetaR)

  force_x = force * np.cos(thetaR) * c
  force_y = force * np.sin(thetaR) * c

  prx_noise = 5*np.random.normal(scale=noise) * np.sqrt(dt**(-1/2)) #so Euler-Maruyama is used to solve SDE in Euler solver function
  pry_noise = 5*np.random.normal(scale=noise) * np.sqrt(dt**(-1/2))
  cbx_noise = 5*np.random.normal(scale=noise) * np.sqrt(dt**(-1/2))
  cby_noise = 5*np.random.normal(scale=noise) * np.sqrt(dt**(-1/2))

  #format of dr is [xpr, ypr, xcb, ycb]
  dr = [(1/nu_pr)*(force_x-F_springx)+prx_noise, (1/nu_pr)*(force_y-F_springy)+pry_noise, (1/nu_cb)*F_springx+cbx_noise, (1/nu_cb)*F_springy+cby_noise]


  return np.array(dr), force, fon_e, foff_e, fon_t, foff_t

###################################################################################################################################################

#Euler solver function that updates cell_motion() function and updates force on or force off events list
#Inputs:
# func => function to solve when ODE is in the form dy/dt = f(t,y) where f(t,y) is func (type: function)
# t_start => time to begin simulation (type: int)
# t_end => end time point to run simulation (type: int)
# dt => step size in solver (type: float)
# y0 => initial condition (in this case starting x and y positions of cell body and protrusion) (type: list of floats)
# force_on_ind0 => initial condition for whether outward driving force is on or off - note: 0 is off and 1 is on (type: ndarray of ints)
# params => list of params for simulation (type: list of floats)
#Outputs:
# T => time points for which solution of ODE was solved at with len(((t_end-t_start)/h)+1) (type: list of floats)
# Y => x and y positions for protrusion and cell body for each time point with (type: ndarray with shape=(len(T),4))
# force_each_step => whether force is on (1) or off (0) at each time step (type: list with len=len(T)-1)
# f_on_events => whether force turned on at each time step (0 means no event, 1 means event) (type: list with len=len(T)-1)
# f_off_events => whether force turned off at each time step (0 means no event, 1 means event) (type: list with len=len(T)-1)
def EulerSolver(func, t_start, t_end, dt, y0, f0, params):
  T = [t_start]
  Y = [y0]

  fon_events = []
  foff_events = []
  fon_times = []
  foff_times = []

  force_each_step = []

  number_steps = int(t_end/dt)


  y_prev = y0
  t_prev = t_start

  f_prev = f0

  time = t_start


  for i in range(number_steps):
    time +=dt
    m, f_curr, fon_e, foff_e, fon_t, foff_t = func(dt, y_prev, f_prev, params,time)
    y_curr = y_prev + dt*m
    t_curr = t_prev + dt
    Y.append(y_curr)
    T.append(t_curr)

    fon_events.append(fon_e)
    foff_events.append(foff_e)
    fon_times.append(fon_t)
    foff_times.append(foff_t)
    force_each_step.append(f_curr)

    y_prev = y_curr
    t_prev = t_curr

    f_prev = f_curr

  return T, np.array(Y), force_each_step, fon_events, foff_events, fon_times, foff_times