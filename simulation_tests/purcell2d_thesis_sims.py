# Generic Simulation Script

import sys
import numpy as np
import matplotlib.pyplot as plt
from integrator_bank import gausss1, gausss2, gausss3, rads2, rads3
from purcell2dfunc import purcell2dfunc
from purcell2djac import purcell2djac

# Solver Setup

# Time array based on time step
dt = .1    # Number of steps
t_max = 500      # Total simulation time
t_arr = np.arange(0.,t_max+dt,dt)

# Time array based on number of steps
# nump = 10000    # Number of steps
# t_max = 10      # Total simulation time
# t_arr, dt= np.linspace(0.,t_max,nump,retstep=True)

# Simulation data container

# Sim Data
gs3simdatalist = np.zeros((t_arr.shape[0],7))


# Initial Data
l = .01                         # Length of each of the links in meters (1 cm)
nu = .95                        # Absolute (Dynamic) viscosity of medium (set to gylcerine)
sr = 10                         # Ratio of length l of link to its cross section radius a (l/a)
kt = (2*np.pi*nu)/np.log(sr)    # Drag coefficient from slender body theory (from Lauga et al.)
params = [l,nu,sr,kt]

# Initial Conditions
# lx,ly,lz = [float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])]
# ang = float(sys.argv[4])*np.pi/8.
# Start in reference position
startvec = np.array([0, np.pi/2., .5*np.pi/180, 
0., .00001, -4.5*.00001, 0*.00001])
# Start in bowl initial conditions
#startvec = np.array([np.pi/4.,np.pi/4., np.cos(ang)*.5*np.pi/180.,np.sin(ang)*.5*np.pi/180., lx,ly,lz])

# Sim add initial conditions
gs3simdatalist[0] = startvec.copy()

# First Step
step = 1

# Sim first step
gs3simdatalist[step] = gausss3(startvec=startvec,params=params,dynfunc=purcell2dfunc,dynjac=purcell2djac,dt=dt) 
print(gs3simdatalist[step])

startvecgs3sim = gs3simdatalist[step]

step += 1

while (step <= t_max/dt):
    # print(step)
    # Sim step
    gs3simdatalist[step] = gausss3(startvec=startvecgs3sim,params=params,dynfunc=purcell2dfunc,dynjac=purcell2djac,dt=dt) 
    print(gs3simdatalist[step])
    startvecgs3sim = gs3simdatalist[step]

    if step%100==0:
            print(step)
    step += 1

# Save data
np.save("purcell2dsim_thesis_data",gs3simdatalist)
# if str(sys.argv[6])=="p":
#     np.save("purcell2d_lxp{}_lyp{}_lzp{}_ang{}_tmax{}_dt001".format(str(lx).split('.')[-1],str(ly).split('.')[-1],str(lz).split('.')[-1],str(int(sys.argv[4])),int(t_max)),gs3simdatalist)
# # For lambda of 1
# elif str(sys.argv[6])=="i":
#     np.save("purcell2d_lx{}_ly{}_lz{}_ang{}_tmax{}_dt001".format(str(int(sys.argv[1])),str(int(sys.argv[2])),str(int(sys.argv[3])),str(int(sys.argv[4])),int(t_max)),gs3simdatalist)
# # For lambda of 1 with offset config
# elif str(sys.argv[6])=="o":
#     np.save("purcell2doff_lx{}_ly{}_lz{}_ang{}_tmax{}_dt001".format(str(int(sys.argv[1])),str(int(sys.argv[2])),str(int(sys.argv[3])),str(int(sys.argv[4])),int(t_max)),gs3simdatalist)

'''
# Load data
data3 = np.load("purcell2d_gausss3_sim_tmax10_dt01.npy")


# phase plot of angles
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4))


ax1.plot(data3[:,0],data3[:,1],'r')
ax2.plot(data3[:,3],data3[:,4],'k')
# ax.set_xlabel('t')
# ax.set_ylabel('l')
plt.show()
'''






