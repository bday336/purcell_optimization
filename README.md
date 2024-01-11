# Numerical Toolkit to Analysis Optimal Control Problem of 3-Link Planar Purcell Microswimmer System

Source code of numerical optimization toolkit for systems possessing a configuration space with a fiber bundle structure. This is an implementation of the process described by Koon and Marsden in [Optimal Control for Holonomic and Nonholonomic Mechanical Systems with Symmetry and Lagrangian Reduction](http://www.cds.caltech.edu/~koon/papers/optimalKM.pdf). Here the question of optimal control is adapted to a variational approach which resembles solving a constrained dynamics problem.

This instance of the toolkit is designed for investigating the particular deformable control system characterizing the process of Low Reynolds Number locomotion performed by microorganisms like the nematodes. This involves cyclic body undulations (wriggling).

The 3-Link Planar Purcell Swimmer Model was first proposed by Purcell in [Life at Low Reynolds Number](https://www.damtp.cam.ac.uk/user/gold/pdfs/purcell.pdf) and has since been the subject of many investigations of dynamics at low Reynolds number. In particular, we use the system setup considered by Ramasamy and Hatton [The Geometry of Optimal Gaits for Drag-Dominated Kinematic Systems](https://ieeexplore.ieee.org/document/8758237/) in terms of forulating the system in terms of fiber bundle structure. We use the energy expenditure to deform as the cost function for our optimization. Given that the Purcell swimmer can only generate translations and rotations in the plane, the symmetry group describing its possible motion is non-abelian (E(2)) and thus approaches like gradient descent are not amenable. We utilize the variational approach here to investigate the question of optimal control with respect to energy expenditure as it can handle systems with non-abelian symmetry groups.