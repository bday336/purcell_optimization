import numpy as np
from src.integrator_files.integrator_bank import gausss1, gausss2, gausss3, rads2, rads3


# Simulation class setup

class PlanarPurcellSystem:
    """
    A class used to simulate optimal trajectory of 3-Link Planar Purcell system

    ...

    Attributes
    ----------
    system_params : array
        the array of parameters describing system consisting of:
            * l  = Length of each link
            * mu = Absolute (Dynamic) viscosity of medium
            * sr = Ratio of length l of link to its cross section radius a (l/a)
            * kt = Drag coefficient from slender body theory (from Lauga et al. 2009)

    dt : float
        the simuation time step size
    tmax : float
        the total simulation time

    solver_id : str
        the solver to be used to evaluate dynamics
        - currently supporting:
            * gs1 = 1-step Gauss collocation 
            * gs2 = 2-step Gauss collocation 
            * gs3 = 3-step Gauss collocation 
            * rs2 = 2-step Radau collocation (RadauIIA)
            * rs3 = 3-step Radau collocation (RadauIIA)

    Methods
    -------
    set_initial_conditions(system_ics)
        Inputs user given initial conditions of system for simulation

    clear_data()
        Clears any simulation data stored in simulation object

    run()
        Runs simulation once given all necessary information

    output_data()
        Outputs simulation data to file with name:
            pmpy_{self.solver_id}_sim_tmax{self.tmax}_dt{self.dt}.npy
    """

    def __init__(self, dyn_function, dyn_jacobian, system_params, dt, tmax, solver_id):
        """
        Parameters
        ----------
        dyn_function : function
            the function describing system (return array)

        dyn_jacobian : function
            the jacobian describing system (return matrix) 
                
        system_params : array
            the array of parameters describing system  

        dt : float
            the simuation time step size
        tmax : float
            the total simulation time

        solver_id : str
            the solver to be used to evaluate dynamics
            - currently supporting:
                * gs1 = 1-step Gauss collocation 
                * gs2 = 2-step Gauss collocation 
                * gs3 = 3-step Gauss collocation 
                * rs2 = 2-step Radau collocation (RadauIIA)
                * rs3 = 3-step Radau collocation (RadauIIA)

        """

        # Function specifying system
        self.dyn_function = dyn_function

        # Jacobian specifying system
        self.dyn_jacobian = dyn_jacobian
        
        # System Parameters
        self.system_params  = system_params

        # Time Data
        self.dt = dt   
        self.tmax = tmax
        self.t_arr = np.arange(0.,self.tmax+self.dt,self.dt)

        # Integrator to use
        self.solver_id = solver_id
        self.tol = 1e-15

        # Internal Flags
        self._have_ics = False
        self._have_run = False

    def set_initial_conditions(self, system_ics):
        """
        Inputs user given initial conditions of system for simulation

        Position and velocity information should be given in terms of the
        parameterization of the ambient space

        Parameters
        ----------
        system_params : array
            the array of parameters describing system consisting of:
                * a1    = initial value of angle 1
                * a2    = initial value of angle 2
                * ad1   = initial velocity of angle 1
                * ad2   = initial velocity of angle 2
                * lamx  = initial value of lagrange multiplier for x constraint
                * lamy  = initial value of lagrange multiplier for y constraint
                * lamth = initial value of lagrange multiplier for theta constraint

        """

        self.system_ics = system_ics
        self._have_ics = True

        # Test System Data
        self.simdatalist = np.zeros((self.t_arr.shape[0],self.system_ics.shape[0]))

    def clear_data(self):
        """
        Clears any simulation data stored in simulation object

        """
        if self._have_ics:
            self.simdatalist = np.zeros((self.t_arr.shape[0],self.system_ics.shape[0]))
            self._have_run = False
        else:
            raise NotImplementedError("Must provide initial conditions via set_initial_conditions(), no data to clear")
        

    def run(self):
        """
        Runs simulation once given all necessary information

        Raises
        ----------
        NotImplementedError
            If no initial conditions have been provided
        """

        if self._have_ics:
            self.simdatalist[0] = self.system_ics.copy()

            # Gauss 1-Step Method
            if self.solver_id == "gs1":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = gausss1(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt       = self.dt,
                        tol      = self.tol)
                    
            # Gauss 2-Step Method
            if self.solver_id == "gs2":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = gausss2(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt       = self.dt,
                        tol      = self.tol)
                    
            # Gauss 3-Step Method
            if self.solver_id == "gs3":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = gausss3(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt       = self.dt,
                        tol      = self.tol)
                    
            # Radau 2-Step Method
            if self.solver_id == "rs2":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = rads2(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt       = self.dt,
                        tol      = self.tol)
                    
            # Radau 3-Step Method
            if self.solver_id == "rs3":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = rads3(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt       = self.dt,
                        tol      = self.tol)

            print("Simulation run completed!")
            self._have_run = True
        else:
            raise NotImplementedError("Must provide initial conditions via set_initial_conditions() before running simulation")

    def output_data(self):
        """
        Outputs simulation data to file with name:
            purcell_{self.solver_id}_sim_tmax{self.tmax}_dt{self.dt}.npy

        Raises
        ----------
        NotImplementedError
            If simulation has not been run, i.e. no data generated
        """

        if self._have_run:
            np.save("purcell_{}_sim_tmax{}_dt{}".format(self.solver_id, str(self.tmax), str(self.dt)), self.simdatalist)
        else:
            raise NotImplementedError("Must use run() to generate data")

    