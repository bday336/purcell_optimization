# Allow for package import
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd())+"/src")

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from src.integrator_files.integrator_bank import gausss1, gausss2, gausss3, rads2, rads3
from src.test_system_simulations.purcell2dfunc import purcell2dfunc
from src.test_system_simulations.purcell2djac import purcell2djac
from src.test_system_simulations.purcell2dj2a import purcell2dj2a
from src.test_system_simulations.PlanarPurcellSystem import PlanarPurcellSystem
from src.function_bank import se2cay, se2group

### Sample Script
### Example simulation of a finding optimal deformation to minimize energy expenditure for Planar Purcell microswimmer



## Time array based on time step
dt = .1          # Time step size
t_max = 10.       # Total simulation time

## System Data (length in units of micrometers)
cell_width  = 100                   # Cell Width (𝜇m) of Nematode
cell_length = 1000                  # Cell Length (𝜇m) of Nematode

link_length = cell_length/3.        # Length of each of the links
nu = .95/(1e6)                      # Absolute (Dynamic) viscosity of medium (set to gylcerine)
sr = cell_length/(.5*cell_width)    # Ratio of length l of link to its cross section radius a (l/a)
kt = (2*np.pi*nu)/np.log(sr)        # Drag coefficient from slender body theory (from Lauga et al.)
params = [link_length,nu,sr,kt]

## Initial Data for System
startvec = np.array([
    0,              # Initial value of angle 1
    np.pi/2.,       # Initial value of angle 2
    .5*np.pi/180*5,   # Initial velocity of angle 1
    0.,             # Initial velocity of angle 2
    .01,         # Initial value of lagrange multiplier for x constraint
    -4.5*.01,    # Initial value of lagrange multiplier for y constraint
    0*.01        # Initial value of lagrange multiplier for theta constraint
    ])


## Solver 
solver_id = "gs1"   # Here using Gauss collocation method with 1 internal step

## Initialize Simulation Object and Run Simulation
sim_test = PlanarPurcellSystem(purcell2dfunc, purcell2djac, params, dt, t_max, solver_id)
sim_test.set_initial_conditions(startvec)
sim_test.run()
sim_test.output_data()


## Read-In Data File for Data Analysis and Visualization
data1 = np.load("purcell_{}_sim_tmax{}_dt{}.npy".format(solver_id, str(t_max), str(dt)))

# --------------------------------------------------------------------
### Plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------

## Plot Space Optimal Trajectory and Corresponding Holonomy
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(16,8))
fig.tight_layout()

## Plot the Optimal Trajectory in Control Space
ax1.plot(data1[:,0],data1[:,1],'r')
ax1.set_xlim(-np.pi,np.pi)
ax1.set_ylim(-np.pi,np.pi)



## Plot the Corresponding Holonomy of Optimal Trajectory

# Calculate the motion along fiber

link2veldat0 = []
caydat0 = []
fulldat0 = []

# Populate data containers
for a in range(data1.shape[0]):
    link2veldat0.append(np.asarray(purcell2dj2a(data1[a,0],data1[a,1],.01)) @ np.array([data1[a,2],data1[a,3]]))
    caydat0.append(se2cay(link2veldat0[-1][0],link2veldat0[-1][1],link2veldat0[-1][2],sim_test.dt))

fulldat0.append(np.identity(3))
for c in range(data1.shape[0]):
    fulldat0.append(fulldat0[-1] @ caydat0[c])

ax2.set_title("x")
ax2.plot(sim_test.t_arr,np.asarray(caydat0)[:,0,2],label="x")
ax3.set_title("y")
ax3.plot(sim_test.t_arr,np.asarray(caydat0)[:,1,2],label="y")
ax4.set_title("th")
ax4.plot(sim_test.t_arr,np.arcsin(np.asarray(caydat0)[:,1,0]),label="th")


plt.show()







