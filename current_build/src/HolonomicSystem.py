import numpy as np
from src.integrator_files.integrator_bank import gausss1, gausss2, gausss3, rads2, rads3
from src.test_system_simulations.test_system_bank import dynfunc_h3simbar, dynjac_h3simbar, dynfunc_s3simbar, dynjac_s3simbar


# Simulation class setup

class HolonomicSystem:
    """
    A class used to perform rody body simulation with spring potential coupling

    ...

    Attributes
    ----------
    ambient_geo : str
        the geometry of ambient space in simulation environment
        - currently supporting:
            * h3   = 3D hyperbolic space
            * s3   = 3D spherical space
            * bump = 3D isotropic nonhomogeneous space (based on https://mathoverflow.net/questions/37651/riemannian-surfaces-with-an-explicit-distance-function)

    system_params : array
        the array of parameters describing system consisting of:
            * v  = initial velocity field
            * ks = stiffness of spring potential
            * x  = rest length of spring potential
            * m  = mass of vertices    

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
            {self.ambient_geo}_r_{self.solver_id}_sim_tmax{self.tmax}_dt{self.dt}.npy
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
                * p1 = initial posiiton of vertex 1
                * p2 = initial posiiton of vertex 2
                * v1 = initial velocity of vertex 1
                * v2 = initial velocity of vertex 2

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
                        dt=self.dt,
                        tol=self.tol)
                    
            # Gauss 2-Step Method
            if self.solver_id == "gs2":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = gausss2(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt=self.dt,
                        tol=self.tol)
                    
            # Gauss 3-Step Method
            if self.solver_id == "gs3":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = gausss3(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt=self.dt,
                        tol=self.tol)
                    
            # Radau 2-Step Method
            if self.solver_id == "rs2":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = rads2(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt=self.dt,
                        tol=self.tol)
                    
            # Radau 3-Step Method
            if self.solver_id == "rs3":
                for step in range(self.t_arr.shape[0] - 1):
                    self.simdatalist[step+1] = rads3(
                        startvec = self.simdatalist[step],
                        params   = self.system_params,
                        dynfunc  = self.dyn_function,
                        dynjac   = self.dyn_jacobian,
                        dt=self.dt,
                        tol=self.tol)

            print("Simulation run completed!")
            self._have_run = True
        else:
            raise NotImplementedError("Must provide initial conditions via set_initial_conditions() before running simulation")

    def output_data(self):
        """
        Outputs simulation data to file with name:
            {self.ambient_geo}_r_{self.solver_id}_sim_tmax{self.tmax}_dt{self.dt}.npy

        Raises
        ----------
        NotImplementedError
            If simulation has not been run, i.e. no data generated
        """

        if self._have_run:
            np.save("{}_sim_tmax{}_dt{}".format(self.solver_id, str(self.tmax), str(self.dt)), self.simdatalist)
        else:
            raise NotImplementedError("Must use run() to generate data")

    