########################
# Integrator Functions #
########################

import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arccos,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,linalg,add


###################
# One Stage Gauss #
###################

def difffuncgauss1s(startvec, params, dynfunc,k1,dt):
    # Set to run Gauss 1-stage method
    a11 = 1./2.

    return np.array([
        k1 - dynfunc(startvec + (a11*k1)*dt, params)
    ]).flatten()

def gausss1(startvec, params, dynfunc, dynjac, dt, tol = 1e-15, imax = 100):
    # Set to run Gauss 1-stage method
    a11 = 1./2.
    bs1 = 1.

    # Initial Guess - Explicit Euler
    k = dynfunc(startvec, params)
    x1guess = startvec + (1./2.)*dt*k
    k1 = dynfunc(x1guess, params)

    # Check Error Before iterations
    er = difffuncgauss1s(startvec, params, dynfunc, k1, dt)

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(er) >= tol and counter <= imax):
        j1 = dynjac(startvec + (a11*k1)*dt, params)
        
        fullj = np.block([
            [np.eye(k.shape[0]) - dt*a11*j1]
        ])

        linsolve = np.linalg.solve(fullj,-er)

        k1 = k1 + linsolve[0:k.shape[0]]

        er = difffuncgauss1s(startvec, params, dynfunc, k1, dt)

        counter += 1

    startvec = startvec + dt*(bs1*k1)
    return startvec.copy()

###################
# Two Stage Gauss #
###################

def difffuncgauss2s(startvec, params, dynfunc,k1,k2,dt):
    # Set to run Gauss 2-stage method
    a11,a12 = [1./4., 1./4. - np.sqrt(3.)/6.]
    a21,a22 = [1./4. + np.sqrt(3.)/6., 1./4.]

    return np.array([
        k1 - dynfunc(startvec + (a11*k1 + a12*k2)*dt, params),
        k2 - dynfunc(startvec + (a21*k1 + a22*k2)*dt, params)
    ]).flatten()

def gausss2(startvec, params, dynfunc, dynjac, dt, tol = 1e-15, imax = 100):
    # Set to run Gauss 2-stage method
    a11,a12 = [1./4., 1./4. - np.sqrt(3.)/6.]
    a21,a22 = [1./4. + np.sqrt(3.)/6., 1./4.]
    bs1,bs2 = [1./2., 1./2.]

    # Initial Guess - Explicit Euler
    k = dynfunc(startvec, params)
    x1guess = startvec + (1./2. - np.sqrt(3.)/6.)*dt*k
    x2guess = startvec + (1./2. + np.sqrt(3.)/6.)*dt*k
    k1 = dynfunc(x1guess, params)
    k2 = dynfunc(x2guess, params)

    # Check Error Before iterations
    er = difffuncgauss2s(startvec, params, dynfunc, k1, k2, dt)

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(er) >= tol and counter <= imax):
        j1 = dynjac(startvec + (a11*k1 + a12*k2)*dt, params)
        j2 = dynjac(startvec + (a21*k1 + a22*k2)*dt, params)
        
        fullj = np.block([
            [np.eye(k.shape[0]) - dt*a11*j1, -(dt*a12*j1)],
            [-(dt*a21*j2), np.eye(k.shape[0]) - dt*a22*j2]
        ])

        linsolve = np.linalg.solve(fullj,-er)

        k1 = k1 + linsolve[0:k.shape[0]]
        k2 = k2 + linsolve[k.shape[0]:2*k.shape[0]]

        er = difffuncgauss2s(startvec, params, dynfunc, k1, k2, dt)

        counter += 1

    startvec = startvec + dt*(bs1*k1 + bs2*k2)
    return startvec.copy()

#####################
# Three Stage Gauss #
#####################

def difffuncgauss3s(startvec, params, dynfunc,k1,k2,k3,dt):
    # Set to run Gauss 3-stage method
    a11,a12,a13 = [5./36., 2./9. - np.sqrt(15.)/15., 5./36. - np.sqrt(15.)/30.]
    a21,a22,a23 = [5./36. + np.sqrt(15.)/24., 2./9., 5./36. - np.sqrt(15.)/24.]
    a31,a32,a33 = [5./36. + np.sqrt(15.)/30., 2./9. + np.sqrt(15.)/15., 5./36.]

    return np.array([
        k1 - dynfunc(startvec + (a11*k1 + a12*k2 + a13*k3)*dt, params),
        k2 - dynfunc(startvec + (a21*k1 + a22*k2 + a23*k3)*dt, params),
        k3 - dynfunc(startvec + (a31*k1 + a32*k2 + a33*k3)*dt, params)
    ]).flatten()

def gausss3(startvec, params, dynfunc, dynjac, dt, tol = 1e-12, imax = 100):
    # Set to run Gauss 3-stage method
    a11,a12,a13 = [5./36., 2./9. - np.sqrt(15.)/15., 5./36. - np.sqrt(15.)/30.]
    a21,a22,a23 = [5./36. + np.sqrt(15.)/24., 2./9., 5./36. - np.sqrt(15.)/24.]
    a31,a32,a33 = [5./36. + np.sqrt(15.)/30., 2./9. + np.sqrt(15.)/15., 5./36.]
    bs1,bs2,bs3 = [5./18., 4./9., 5./18.]

    # Initial Guess - Explicit Euler
    k = dynfunc(startvec, params)
    x1guess = startvec + (1./2. - np.sqrt(15.)/10.)*dt*k
    x2guess = startvec + (1./2.)*dt*k
    x3guess = startvec + (1./2. + np.sqrt(15.)/10.)*dt*k
    k1 = dynfunc(x1guess, params)
    k2 = dynfunc(x2guess, params)
    k3 = dynfunc(x3guess, params)

    # Check Error Before iterations
    er = difffuncgauss3s(startvec, params, dynfunc, k1, k2, k3, dt)

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(er) >= tol and counter <= imax):
        j1 = dynjac(startvec + (a11*k1 + a12*k2 + a13*k3)*dt, params)
        j2 = dynjac(startvec + (a21*k1 + a22*k2 + a23*k3)*dt, params)
        j3 = dynjac(startvec + (a31*k1 + a32*k2 + a33*k3)*dt, params)
        
        fullj = np.block([
            [np.eye(k.shape[0]) - dt*a11*j1, -(dt*a12*j1), -(dt*a13*j1)],
            [-(dt*a21*j2), np.eye(k.shape[0]) - dt*a22*j2, -(dt*a23*j2)],
            [-(dt*a31*j3), -(dt*a32*j3), np.eye(k.shape[0]) - dt*a33*j3]
        ])

        linsolve = np.linalg.solve(fullj,-er)

        k1 = k1 + linsolve[0:k.shape[0]]
        k2 = k2 + linsolve[k.shape[0]:2*k.shape[0]]
        k3 = k3 + linsolve[2*k.shape[0]:3*k.shape[0]]

        er = difffuncgauss3s(startvec, params, dynfunc, k1, k2, k3, dt)

        counter += 1

    startvec = startvec + dt*(bs1*k1 + bs2*k2 + bs3*k3)
    return startvec.copy()

###################
# Two Stage Radau #
###################

def difffuncrad2s(startvec, params, dynfunc,k1,k2,dt):
    # Set to run RadauIIA 2-stage method
    a11,a12 = [5./12., -1./12.]
    a21,a22 = [3./4., 1./4.]

    return np.array([
        k1 - dynfunc(startvec + (a11*k1 + a12*k2)*dt, params),
        k2 - dynfunc(startvec + (a21*k1 + a22*k2)*dt, params)
    ]).flatten()

def rads2(startvec, params, dynfunc, dynjac, dt, tol = 1e-15, imax = 100):
    # Set to run RadauIIA 2-stage method
    a11,a12 = [5./12., -1./12.]
    a21,a22 = [3./4., 1./4.]
    bs1,bs2 = [3./4., 1./4.]

    # Initial Guess - Explicit Euler
    k = dynfunc(startvec, params)
    x1guess = startvec + (1./3.)*dt*k
    x2guess = startvec + (1.)*dt*k
    k1 = dynfunc(x1guess, params)
    k2 = dynfunc(x2guess, params)

    # Check Error Before iterations
    er = difffuncrad2s(startvec, params, dynfunc, k1, k2, dt)

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(er) >= tol and counter <= imax):
        j1 = dynjac(startvec + (a11*k1 + a12*k2)*dt, params)
        j2 = dynjac(startvec + (a21*k1 + a22*k2)*dt, params)
        
        fullj = np.block([
            [np.eye(k.shape[0]) - dt*a11*j1, -(dt*a12*j1)],
            [-(dt*a21*j2), np.eye(k.shape[0]) - dt*a22*j2]
        ])

        linsolve = np.linalg.solve(fullj,-er)

        k1 = k1 + linsolve[0:k.shape[0]]
        k2 = k2 + linsolve[k.shape[0]:2*k.shape[0]]

        er = difffuncrad2s(startvec, params, dynfunc, k1, k2, dt)

        counter += 1

    startvec = startvec + dt*(bs1*k1 + bs2*k2)
    return startvec.copy()

#####################
# Three Stage Radau #
#####################

def difffuncrad3s(startvec, params, dynfunc,k1,k2,k3,dt):
    root6 = np.sqrt(6.)
    # Set to run RadauIIA 3-stage method
    a11,a12,a13 = [(88.-7.*root6)/360., (296.-169.*root6)/1800., (-2.+3.*root6)/225.]
    a21,a22,a23 = [(296.+169.*root6)/1800., (88.+7.*root6)/360., (-2.-3.*root6)/225.]
    a31,a32,a33 = [(16.-root6)/36., (16.+root6)/36., 1./9.]

    return np.array([
        k1 - dynfunc(startvec + (a11*k1 + a12*k2 + a13*k3)*dt, params),
        k2 - dynfunc(startvec + (a21*k1 + a22*k2 + a23*k3)*dt, params),
        k3 - dynfunc(startvec + (a31*k1 + a32*k2 + a33*k3)*dt, params)
    ]).flatten()

def rads3(startvec, params, dynfunc, dynjac, dt, tol = 1e-15, imax = 100):
    root6 = np.sqrt(6.)
    # Set to run RadauIIA 3-stage method
    a11,a12,a13 = [(88.-7.*root6)/360., (296.-169.*root6)/1800., (-2.+3.*root6)/225.]
    a21,a22,a23 = [(296.+169.*root6)/1800., (88.+7.*root6)/360., (-2.-3.*root6)/225.]
    a31,a32,a33 = [(16.-root6)/36., (16.+root6)/36., 1./9.]
    bs1,bs2,bs3 = [(16.-root6)/36., (16.+root6)/36., 1./9.]

    # Initial Guess - Explicit Euler
    k = dynfunc(startvec, params)
    x1guess = startvec + ((4.-root6)/10.)*dt*k
    x2guess = startvec + ((4.+root6)/10.)*dt*k
    x3guess = startvec + (1.)*dt*k
    k1 = dynfunc(x1guess, params)
    k2 = dynfunc(x2guess, params)
    k3 = dynfunc(x3guess, params)

    # Check Error Before iterations
    er = difffuncrad3s(startvec, params, dynfunc, k1, k2, k3, dt)

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(er) >= tol and counter <= imax):
        j1 = dynjac(startvec + (a11*k1 + a12*k2 + a13*k3)*dt, params)
        j2 = dynjac(startvec + (a21*k1 + a22*k2 + a23*k3)*dt, params)
        j3 = dynjac(startvec + (a31*k1 + a32*k2 + a33*k3)*dt, params)
        
        fullj = np.block([
            [np.eye(k.shape[0]) - dt*a11*j1, -(dt*a12*j1), -(dt*a13*j1)],
            [-(dt*a21*j2), np.eye(k.shape[0]) - dt*a22*j2, -(dt*a23*j2)],
            [-(dt*a31*j3), -(dt*a32*j3), np.eye(k.shape[0]) - dt*a33*j3]
        ])

        linsolve = np.linalg.solve(fullj,-er)

        k1 = k1 + linsolve[0:k.shape[0]]
        k2 = k2 + linsolve[k.shape[0]:2*k.shape[0]]
        k3 = k3 + linsolve[2*k.shape[0]:3*k.shape[0]]

        er = difffuncrad3s(startvec, params, dynfunc, k1, k2, k3, dt)

        counter += 1

    startvec = startvec + dt*(bs1*k1 + bs2*k2 + bs3*k3)
    return startvec.copy()

