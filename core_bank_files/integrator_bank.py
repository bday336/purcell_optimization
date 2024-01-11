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



#####################
# Rigid Rod Solvers #
#####################


####################################
# Two Stage Radau (Rigid DAE - H3) #
####################################

# This is the manual solver - it seems to work

def h3rads2roddae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],
        stepc1[1],stepc2[1],
        stepc1[2],stepc2[2],

        stepc1[3],stepc2[3],
        stepc1[4],stepc2[4],
        stepc1[5],stepc2[5],

        stepc1[6],stepc2[6],
        stepc1[7],stepc2[7],
        stepc1[8],stepc2[8],

        stepc1[9],stepc2[9],
        stepc1[10],stepc2[10],
        stepc1[11],stepc2[11],

        stepc1[12],stepc2[12]
        ])

    ex1,dex1 = h3rads2rodex1(stepn,stepc1,stepc2,dt,params)
    ex2,dex2 = h3rads2rodex2(stepn,stepc1,stepc2,dt,params)
    ex3,dex3 = h3rads2rodex3(stepn,stepc1,stepc2,dt,params)
    ex4,dex4 = h3rads2rodex4(stepn,stepc1,stepc2,dt,params)
    ex5,dex5 = h3rads2rodex5(stepn,stepc1,stepc2,dt,params)
    ex6,dex6 = h3rads2rodex6(stepn,stepc1,stepc2,dt,params)

    ex7,dex7 = h3rads2rodex7(stepn,stepc1,stepc2,dt,params)
    ex8,dex8 = h3rads2rodex8(stepn,stepc1,stepc2,dt,params)
    ex9,dex9 = h3rads2rodex9(stepn,stepc1,stepc2,dt,params)
    ex10,dex10 = h3rads2rodex10(stepn,stepc1,stepc2,dt,params)
    ex11,dex11 = h3rads2rodex11(stepn,stepc1,stepc2,dt,params)
    ex12,dex12 = h3rads2rodex12(stepn,stepc1,stepc2,dt,params)

    ex13,dex13 = h3rads2rodex13(stepn,stepc1,stepc2,dt,params)
    ex14,dex14 = h3rads2rodex14(stepn,stepc1,stepc2,dt,params)
    ex15,dex15 = h3rads2rodex15(stepn,stepc1,stepc2,dt,params)
    ex16,dex16 = h3rads2rodex16(stepn,stepc1,stepc2,dt,params)
    ex17,dex17 = h3rads2rodex17(stepn,stepc1,stepc2,dt,params)
    ex18,dex18 = h3rads2rodex18(stepn,stepc1,stepc2,dt,params)

    ex19,dex19 = h3rads2rodex19(stepn,stepc1,stepc2,dt,params)
    ex20,dex20 = h3rads2rodex20(stepn,stepc1,stepc2,dt,params)
    ex21,dex21 = h3rads2rodex21(stepn,stepc1,stepc2,dt,params)
    ex22,dex22 = h3rads2rodex22(stepn,stepc1,stepc2,dt,params)
    ex23,dex23 = h3rads2rodex23(stepn,stepc1,stepc2,dt,params)
    ex24,dex24 = h3rads2rodex24(stepn,stepc1,stepc2,dt,params)

    ex25,dex25 = h3rads2rodex25(stepn,stepc1,stepc2,dt,params)
    ex26,dex26 = h3rads2rodex26(stepn,stepc1,stepc2,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,

        dex7,dex8,dex9,
        dex10,dex11,dex12,

        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,

        dex25,dex26

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,

        ex7,ex8,ex9,
        ex10,ex11,ex12,

        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,

        ex25,ex26
    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::2].copy()
        stepc2 = val1[1::2].copy()

        ex1,dex1 = h3rads2rodex1(stepn,stepc1,stepc2,dt,params)
        ex2,dex2 = h3rads2rodex2(stepn,stepc1,stepc2,dt,params)
        ex3,dex3 = h3rads2rodex3(stepn,stepc1,stepc2,dt,params)
        ex4,dex4 = h3rads2rodex4(stepn,stepc1,stepc2,dt,params)
        ex5,dex5 = h3rads2rodex5(stepn,stepc1,stepc2,dt,params)
        ex6,dex6 = h3rads2rodex6(stepn,stepc1,stepc2,dt,params)

        ex7,dex7 = h3rads2rodex7(stepn,stepc1,stepc2,dt,params)
        ex8,dex8 = h3rads2rodex8(stepn,stepc1,stepc2,dt,params)
        ex9,dex9 = h3rads2rodex9(stepn,stepc1,stepc2,dt,params)
        ex10,dex10 = h3rads2rodex10(stepn,stepc1,stepc2,dt,params)
        ex11,dex11 = h3rads2rodex11(stepn,stepc1,stepc2,dt,params)
        ex12,dex12 = h3rads2rodex12(stepn,stepc1,stepc2,dt,params)

        ex13,dex13 = h3rads2rodex13(stepn,stepc1,stepc2,dt,params)
        ex14,dex14 = h3rads2rodex14(stepn,stepc1,stepc2,dt,params)
        ex15,dex15 = h3rads2rodex15(stepn,stepc1,stepc2,dt,params)
        ex16,dex16 = h3rads2rodex16(stepn,stepc1,stepc2,dt,params)
        ex17,dex17 = h3rads2rodex17(stepn,stepc1,stepc2,dt,params)
        ex18,dex18 = h3rads2rodex18(stepn,stepc1,stepc2,dt,params)

        ex19,dex19 = h3rads2rodex19(stepn,stepc1,stepc2,dt,params)
        ex20,dex20 = h3rads2rodex20(stepn,stepc1,stepc2,dt,params)
        ex21,dex21 = h3rads2rodex21(stepn,stepc1,stepc2,dt,params)
        ex22,dex22 = h3rads2rodex22(stepn,stepc1,stepc2,dt,params)
        ex23,dex23 = h3rads2rodex23(stepn,stepc1,stepc2,dt,params)
        ex24,dex24 = h3rads2rodex24(stepn,stepc1,stepc2,dt,params)

        ex25,dex25 = h3rads2rodex25(stepn,stepc1,stepc2,dt,params)
        ex26,dex26 = h3rads2rodex26(stepn,stepc1,stepc2,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,

            dex7,dex8,dex9,
            dex10,dex11,dex12,

            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,

            dex25,dex26

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,

            ex7,ex8,ex9,
            ex10,ex11,ex12,

            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,

            ex25,ex26
        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[1::2]

######################################
# Three Stage Radau (Rigid DAE - H3) #
######################################

# This is the manual solver - it seems to work

def h3rads3roddae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    stepn1 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],stepn1[0],
        stepc1[1],stepc2[1],stepn1[1],
        stepc1[2],stepc2[2],stepn1[2],

        stepc1[3],stepc2[3],stepn1[3],
        stepc1[4],stepc2[4],stepn1[4],
        stepc1[5],stepc2[5],stepn1[5],

        stepc1[6],stepc2[6],stepn1[6],
        stepc1[7],stepc2[7],stepn1[7],
        stepc1[8],stepc2[8],stepn1[8],

        stepc1[9],stepc2[9],stepn1[9],
        stepc1[10],stepc2[10],stepn1[10],
        stepc1[11],stepc2[11],stepn1[11],

        stepc1[12],stepc2[12],stepn1[12]
        ])

    ex1,dex1 = h3rads3rodex1(stepn,stepc1,stepc2,stepn1,dt,params)
    ex2,dex2 = h3rads3rodex2(stepn,stepc1,stepc2,stepn1,dt,params)
    ex3,dex3 = h3rads3rodex3(stepn,stepc1,stepc2,stepn1,dt,params)
    ex4,dex4 = h3rads3rodex4(stepn,stepc1,stepc2,stepn1,dt,params)
    ex5,dex5 = h3rads3rodex5(stepn,stepc1,stepc2,stepn1,dt,params)
    ex6,dex6 = h3rads3rodex6(stepn,stepc1,stepc2,stepn1,dt,params)
    ex7,dex7 = h3rads3rodex7(stepn,stepc1,stepc2,stepn1,dt,params)
    ex8,dex8 = h3rads3rodex8(stepn,stepc1,stepc2,stepn1,dt,params)
    ex9,dex9 = h3rads3rodex9(stepn,stepc1,stepc2,stepn1,dt,params)

    ex10,dex10 = h3rads3rodex10(stepn,stepc1,stepc2,stepn1,dt,params)
    ex11,dex11 = h3rads3rodex11(stepn,stepc1,stepc2,stepn1,dt,params)
    ex12,dex12 = h3rads3rodex12(stepn,stepc1,stepc2,stepn1,dt,params)
    ex13,dex13 = h3rads3rodex13(stepn,stepc1,stepc2,stepn1,dt,params)
    ex14,dex14 = h3rads3rodex14(stepn,stepc1,stepc2,stepn1,dt,params)
    ex15,dex15 = h3rads3rodex15(stepn,stepc1,stepc2,stepn1,dt,params)
    ex16,dex16 = h3rads3rodex16(stepn,stepc1,stepc2,stepn1,dt,params)
    ex17,dex17 = h3rads3rodex17(stepn,stepc1,stepc2,stepn1,dt,params)
    ex18,dex18 = h3rads3rodex18(stepn,stepc1,stepc2,stepn1,dt,params)

    ex19,dex19 = h3rads3rodex19(stepn,stepc1,stepc2,stepn1,dt,params)
    ex20,dex20 = h3rads3rodex20(stepn,stepc1,stepc2,stepn1,dt,params)
    ex21,dex21 = h3rads3rodex21(stepn,stepc1,stepc2,stepn1,dt,params)
    ex22,dex22 = h3rads3rodex22(stepn,stepc1,stepc2,stepn1,dt,params)
    ex23,dex23 = h3rads3rodex23(stepn,stepc1,stepc2,stepn1,dt,params)
    ex24,dex24 = h3rads3rodex24(stepn,stepc1,stepc2,stepn1,dt,params)
    ex25,dex25 = h3rads3rodex25(stepn,stepc1,stepc2,stepn1,dt,params)
    ex26,dex26 = h3rads3rodex26(stepn,stepc1,stepc2,stepn1,dt,params)
    ex27,dex27 = h3rads3rodex27(stepn,stepc1,stepc2,stepn1,dt,params)

    ex28,dex28 = h3rads3rodex28(stepn,stepc1,stepc2,stepn1,dt,params)
    ex29,dex29 = h3rads3rodex29(stepn,stepc1,stepc2,stepn1,dt,params)
    ex30,dex30 = h3rads3rodex30(stepn,stepc1,stepc2,stepn1,dt,params)
    ex31,dex31 = h3rads3rodex31(stepn,stepc1,stepc2,stepn1,dt,params)
    ex32,dex32 = h3rads3rodex32(stepn,stepc1,stepc2,stepn1,dt,params)
    ex33,dex33 = h3rads3rodex33(stepn,stepc1,stepc2,stepn1,dt,params)
    ex34,dex34 = h3rads3rodex34(stepn,stepc1,stepc2,stepn1,dt,params)
    ex35,dex35 = h3rads3rodex35(stepn,stepc1,stepc2,stepn1,dt,params)
    ex36,dex36 = h3rads3rodex36(stepn,stepc1,stepc2,stepn1,dt,params)

    ex37,dex37 = h3rads3rodex37(stepn,stepc1,stepc2,stepn1,dt,params)
    ex38,dex38 = h3rads3rodex38(stepn,stepc1,stepc2,stepn1,dt,params)
    ex39,dex39 = h3rads3rodex39(stepn,stepc1,stepc2,stepn1,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,
        dex7,dex8,dex9,

        dex10,dex11,dex12,
        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,
        dex25,dex26,dex27,

        dex28,dex29,dex30,
        dex31,dex32,dex33,
        dex34,dex35,dex36,

        dex37,dex38,dex39

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,
        ex7,ex8,ex9,

        ex10,ex11,ex12,
        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,
        ex25,ex26,ex27,

        ex28,ex29,ex30,
        ex31,ex32,ex33,
        ex34,ex35,ex36,
        
        ex37,ex38,ex39
    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::3].copy()
        stepc2 = val1[1::3].copy()
        stepn1 = val1[2::3].copy()

        ex1,dex1 = h3rads3rodex1(stepn,stepc1,stepc2,stepn1,dt,params)
        ex2,dex2 = h3rads3rodex2(stepn,stepc1,stepc2,stepn1,dt,params)
        ex3,dex3 = h3rads3rodex3(stepn,stepc1,stepc2,stepn1,dt,params)
        ex4,dex4 = h3rads3rodex4(stepn,stepc1,stepc2,stepn1,dt,params)
        ex5,dex5 = h3rads3rodex5(stepn,stepc1,stepc2,stepn1,dt,params)
        ex6,dex6 = h3rads3rodex6(stepn,stepc1,stepc2,stepn1,dt,params)
        ex7,dex7 = h3rads3rodex7(stepn,stepc1,stepc2,stepn1,dt,params)
        ex8,dex8 = h3rads3rodex8(stepn,stepc1,stepc2,stepn1,dt,params)
        ex9,dex9 = h3rads3rodex9(stepn,stepc1,stepc2,stepn1,dt,params)

        ex10,dex10 = h3rads3rodex10(stepn,stepc1,stepc2,stepn1,dt,params)
        ex11,dex11 = h3rads3rodex11(stepn,stepc1,stepc2,stepn1,dt,params)
        ex12,dex12 = h3rads3rodex12(stepn,stepc1,stepc2,stepn1,dt,params)
        ex13,dex13 = h3rads3rodex13(stepn,stepc1,stepc2,stepn1,dt,params)
        ex14,dex14 = h3rads3rodex14(stepn,stepc1,stepc2,stepn1,dt,params)
        ex15,dex15 = h3rads3rodex15(stepn,stepc1,stepc2,stepn1,dt,params)
        ex16,dex16 = h3rads3rodex16(stepn,stepc1,stepc2,stepn1,dt,params)
        ex17,dex17 = h3rads3rodex17(stepn,stepc1,stepc2,stepn1,dt,params)
        ex18,dex18 = h3rads3rodex18(stepn,stepc1,stepc2,stepn1,dt,params)

        ex19,dex19 = h3rads3rodex19(stepn,stepc1,stepc2,stepn1,dt,params)
        ex20,dex20 = h3rads3rodex20(stepn,stepc1,stepc2,stepn1,dt,params)
        ex21,dex21 = h3rads3rodex21(stepn,stepc1,stepc2,stepn1,dt,params)
        ex22,dex22 = h3rads3rodex22(stepn,stepc1,stepc2,stepn1,dt,params)
        ex23,dex23 = h3rads3rodex23(stepn,stepc1,stepc2,stepn1,dt,params)
        ex24,dex24 = h3rads3rodex24(stepn,stepc1,stepc2,stepn1,dt,params)
        ex25,dex25 = h3rads3rodex25(stepn,stepc1,stepc2,stepn1,dt,params)
        ex26,dex26 = h3rads3rodex26(stepn,stepc1,stepc2,stepn1,dt,params)
        ex27,dex27 = h3rads3rodex27(stepn,stepc1,stepc2,stepn1,dt,params)

        ex28,dex28 = h3rads3rodex28(stepn,stepc1,stepc2,stepn1,dt,params)
        ex29,dex29 = h3rads3rodex29(stepn,stepc1,stepc2,stepn1,dt,params)
        ex30,dex30 = h3rads3rodex30(stepn,stepc1,stepc2,stepn1,dt,params)
        ex31,dex31 = h3rads3rodex31(stepn,stepc1,stepc2,stepn1,dt,params)
        ex32,dex32 = h3rads3rodex32(stepn,stepc1,stepc2,stepn1,dt,params)
        ex33,dex33 = h3rads3rodex33(stepn,stepc1,stepc2,stepn1,dt,params)
        ex34,dex34 = h3rads3rodex34(stepn,stepc1,stepc2,stepn1,dt,params)
        ex35,dex35 = h3rads3rodex35(stepn,stepc1,stepc2,stepn1,dt,params)
        ex36,dex36 = h3rads3rodex36(stepn,stepc1,stepc2,stepn1,dt,params)

        ex37,dex37 = h3rads3rodex37(stepn,stepc1,stepc2,stepn1,dt,params)
        ex38,dex38 = h3rads3rodex38(stepn,stepc1,stepc2,stepn1,dt,params)
        ex39,dex39 = h3rads3rodex39(stepn,stepc1,stepc2,stepn1,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,
            dex7,dex8,dex9,

            dex10,dex11,dex12,
            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,
            dex25,dex26,dex27,

            dex28,dex29,dex30,
            dex31,dex32,dex33,
            dex34,dex35,dex36,

            dex37,dex38,dex39

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,
            ex7,ex8,ex9,

            ex10,ex11,ex12,
            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,
            ex25,ex26,ex27,

            ex28,ex29,ex30,
            ex31,ex32,ex33,
            ex34,ex35,ex36,
            
            ex37,ex38,ex39
        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[2::3]


####################################
# Two Stage Radau (Rigid DAE - S3) #
####################################

# This is the manual solver - it seems to work

def s3rads2roddae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],
        stepc1[1],stepc2[1],
        stepc1[2],stepc2[2],

        stepc1[3],stepc2[3],
        stepc1[4],stepc2[4],
        stepc1[5],stepc2[5],

        stepc1[6],stepc2[6],
        stepc1[7],stepc2[7],
        stepc1[8],stepc2[8],

        stepc1[9],stepc2[9],
        stepc1[10],stepc2[10],
        stepc1[11],stepc2[11],

        stepc1[12],stepc2[12]
        ])

    ex1,dex1 = s3rads2rodex1(stepn,stepc1,stepc2,dt,params)
    ex2,dex2 = s3rads2rodex2(stepn,stepc1,stepc2,dt,params)
    ex3,dex3 = s3rads2rodex3(stepn,stepc1,stepc2,dt,params)
    ex4,dex4 = s3rads2rodex4(stepn,stepc1,stepc2,dt,params)
    ex5,dex5 = s3rads2rodex5(stepn,stepc1,stepc2,dt,params)
    ex6,dex6 = s3rads2rodex6(stepn,stepc1,stepc2,dt,params)

    ex7,dex7 = s3rads2rodex7(stepn,stepc1,stepc2,dt,params)
    ex8,dex8 = s3rads2rodex8(stepn,stepc1,stepc2,dt,params)
    ex9,dex9 = s3rads2rodex9(stepn,stepc1,stepc2,dt,params)
    ex10,dex10 = s3rads2rodex10(stepn,stepc1,stepc2,dt,params)
    ex11,dex11 = s3rads2rodex11(stepn,stepc1,stepc2,dt,params)
    ex12,dex12 = s3rads2rodex12(stepn,stepc1,stepc2,dt,params)

    ex13,dex13 = s3rads2rodex13(stepn,stepc1,stepc2,dt,params)
    ex14,dex14 = s3rads2rodex14(stepn,stepc1,stepc2,dt,params)
    ex15,dex15 = s3rads2rodex15(stepn,stepc1,stepc2,dt,params)
    ex16,dex16 = s3rads2rodex16(stepn,stepc1,stepc2,dt,params)
    ex17,dex17 = s3rads2rodex17(stepn,stepc1,stepc2,dt,params)
    ex18,dex18 = s3rads2rodex18(stepn,stepc1,stepc2,dt,params)

    ex19,dex19 = s3rads2rodex19(stepn,stepc1,stepc2,dt,params)
    ex20,dex20 = s3rads2rodex20(stepn,stepc1,stepc2,dt,params)
    ex21,dex21 = s3rads2rodex21(stepn,stepc1,stepc2,dt,params)
    ex22,dex22 = s3rads2rodex22(stepn,stepc1,stepc2,dt,params)
    ex23,dex23 = s3rads2rodex23(stepn,stepc1,stepc2,dt,params)
    ex24,dex24 = s3rads2rodex24(stepn,stepc1,stepc2,dt,params)

    ex25,dex25 = s3rads2rodex25(stepn,stepc1,stepc2,dt,params)
    ex26,dex26 = s3rads2rodex26(stepn,stepc1,stepc2,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,

        dex7,dex8,dex9,
        dex10,dex11,dex12,

        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,

        dex25,dex26

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,

        ex7,ex8,ex9,
        ex10,ex11,ex12,

        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,

        ex25,ex26
    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::2].copy()
        stepc2 = val1[1::2].copy()

        ex1,dex1 = s3rads2rodex1(stepn,stepc1,stepc2,dt,params)
        ex2,dex2 = s3rads2rodex2(stepn,stepc1,stepc2,dt,params)
        ex3,dex3 = s3rads2rodex3(stepn,stepc1,stepc2,dt,params)
        ex4,dex4 = s3rads2rodex4(stepn,stepc1,stepc2,dt,params)
        ex5,dex5 = s3rads2rodex5(stepn,stepc1,stepc2,dt,params)
        ex6,dex6 = s3rads2rodex6(stepn,stepc1,stepc2,dt,params)

        ex7,dex7 = s3rads2rodex7(stepn,stepc1,stepc2,dt,params)
        ex8,dex8 = s3rads2rodex8(stepn,stepc1,stepc2,dt,params)
        ex9,dex9 = s3rads2rodex9(stepn,stepc1,stepc2,dt,params)
        ex10,dex10 = s3rads2rodex10(stepn,stepc1,stepc2,dt,params)
        ex11,dex11 = s3rads2rodex11(stepn,stepc1,stepc2,dt,params)
        ex12,dex12 = s3rads2rodex12(stepn,stepc1,stepc2,dt,params)

        ex13,dex13 = s3rads2rodex13(stepn,stepc1,stepc2,dt,params)
        ex14,dex14 = s3rads2rodex14(stepn,stepc1,stepc2,dt,params)
        ex15,dex15 = s3rads2rodex15(stepn,stepc1,stepc2,dt,params)
        ex16,dex16 = s3rads2rodex16(stepn,stepc1,stepc2,dt,params)
        ex17,dex17 = s3rads2rodex17(stepn,stepc1,stepc2,dt,params)
        ex18,dex18 = s3rads2rodex18(stepn,stepc1,stepc2,dt,params)

        ex19,dex19 = s3rads2rodex19(stepn,stepc1,stepc2,dt,params)
        ex20,dex20 = s3rads2rodex20(stepn,stepc1,stepc2,dt,params)
        ex21,dex21 = s3rads2rodex21(stepn,stepc1,stepc2,dt,params)
        ex22,dex22 = s3rads2rodex22(stepn,stepc1,stepc2,dt,params)
        ex23,dex23 = s3rads2rodex23(stepn,stepc1,stepc2,dt,params)
        ex24,dex24 = s3rads2rodex24(stepn,stepc1,stepc2,dt,params)

        ex25,dex25 = s3rads2rodex25(stepn,stepc1,stepc2,dt,params)
        ex26,dex26 = s3rads2rodex26(stepn,stepc1,stepc2,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,

            dex7,dex8,dex9,
            dex10,dex11,dex12,

            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,

            dex25,dex26

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,

            ex7,ex8,ex9,
            ex10,ex11,ex12,

            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,

            ex25,ex26
        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[1::2]

######################################
# Three Stage Radau (Rigid DAE - S3) #
######################################

# This is the manual solver - it seems to work

def s3rads3roddae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    stepn1 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],stepn1[0],
        stepc1[1],stepc2[1],stepn1[1],
        stepc1[2],stepc2[2],stepn1[2],

        stepc1[3],stepc2[3],stepn1[3],
        stepc1[4],stepc2[4],stepn1[4],
        stepc1[5],stepc2[5],stepn1[5],

        stepc1[6],stepc2[6],stepn1[6],
        stepc1[7],stepc2[7],stepn1[7],
        stepc1[8],stepc2[8],stepn1[8],

        stepc1[9],stepc2[9],stepn1[9],
        stepc1[10],stepc2[10],stepn1[10],
        stepc1[11],stepc2[11],stepn1[11],

        stepc1[12],stepc2[12],stepn1[12]
        ])

    ex1,dex1 = s3rads3rodex1(stepn,stepc1,stepc2,stepn1,dt,params)
    ex2,dex2 = s3rads3rodex2(stepn,stepc1,stepc2,stepn1,dt,params)
    ex3,dex3 = s3rads3rodex3(stepn,stepc1,stepc2,stepn1,dt,params)
    ex4,dex4 = s3rads3rodex4(stepn,stepc1,stepc2,stepn1,dt,params)
    ex5,dex5 = s3rads3rodex5(stepn,stepc1,stepc2,stepn1,dt,params)
    ex6,dex6 = s3rads3rodex6(stepn,stepc1,stepc2,stepn1,dt,params)
    ex7,dex7 = s3rads3rodex7(stepn,stepc1,stepc2,stepn1,dt,params)
    ex8,dex8 = s3rads3rodex8(stepn,stepc1,stepc2,stepn1,dt,params)
    ex9,dex9 = s3rads3rodex9(stepn,stepc1,stepc2,stepn1,dt,params)

    ex10,dex10 = s3rads3rodex10(stepn,stepc1,stepc2,stepn1,dt,params)
    ex11,dex11 = s3rads3rodex11(stepn,stepc1,stepc2,stepn1,dt,params)
    ex12,dex12 = s3rads3rodex12(stepn,stepc1,stepc2,stepn1,dt,params)
    ex13,dex13 = s3rads3rodex13(stepn,stepc1,stepc2,stepn1,dt,params)
    ex14,dex14 = s3rads3rodex14(stepn,stepc1,stepc2,stepn1,dt,params)
    ex15,dex15 = s3rads3rodex15(stepn,stepc1,stepc2,stepn1,dt,params)
    ex16,dex16 = s3rads3rodex16(stepn,stepc1,stepc2,stepn1,dt,params)
    ex17,dex17 = s3rads3rodex17(stepn,stepc1,stepc2,stepn1,dt,params)
    ex18,dex18 = s3rads3rodex18(stepn,stepc1,stepc2,stepn1,dt,params)

    ex19,dex19 = s3rads3rodex19(stepn,stepc1,stepc2,stepn1,dt,params)
    ex20,dex20 = s3rads3rodex20(stepn,stepc1,stepc2,stepn1,dt,params)
    ex21,dex21 = s3rads3rodex21(stepn,stepc1,stepc2,stepn1,dt,params)
    ex22,dex22 = s3rads3rodex22(stepn,stepc1,stepc2,stepn1,dt,params)
    ex23,dex23 = s3rads3rodex23(stepn,stepc1,stepc2,stepn1,dt,params)
    ex24,dex24 = s3rads3rodex24(stepn,stepc1,stepc2,stepn1,dt,params)
    ex25,dex25 = s3rads3rodex25(stepn,stepc1,stepc2,stepn1,dt,params)
    ex26,dex26 = s3rads3rodex26(stepn,stepc1,stepc2,stepn1,dt,params)
    ex27,dex27 = s3rads3rodex27(stepn,stepc1,stepc2,stepn1,dt,params)

    ex28,dex28 = s3rads3rodex28(stepn,stepc1,stepc2,stepn1,dt,params)
    ex29,dex29 = s3rads3rodex29(stepn,stepc1,stepc2,stepn1,dt,params)
    ex30,dex30 = s3rads3rodex30(stepn,stepc1,stepc2,stepn1,dt,params)
    ex31,dex31 = s3rads3rodex31(stepn,stepc1,stepc2,stepn1,dt,params)
    ex32,dex32 = s3rads3rodex32(stepn,stepc1,stepc2,stepn1,dt,params)
    ex33,dex33 = s3rads3rodex33(stepn,stepc1,stepc2,stepn1,dt,params)
    ex34,dex34 = s3rads3rodex34(stepn,stepc1,stepc2,stepn1,dt,params)
    ex35,dex35 = s3rads3rodex35(stepn,stepc1,stepc2,stepn1,dt,params)
    ex36,dex36 = s3rads3rodex36(stepn,stepc1,stepc2,stepn1,dt,params)

    ex37,dex37 = s3rads3rodex37(stepn,stepc1,stepc2,stepn1,dt,params)
    ex38,dex38 = s3rads3rodex38(stepn,stepc1,stepc2,stepn1,dt,params)
    ex39,dex39 = s3rads3rodex39(stepn,stepc1,stepc2,stepn1,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,
        dex7,dex8,dex9,

        dex10,dex11,dex12,
        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,
        dex25,dex26,dex27,

        dex28,dex29,dex30,
        dex31,dex32,dex33,
        dex34,dex35,dex36,

        dex37,dex38,dex39

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,
        ex7,ex8,ex9,

        ex10,ex11,ex12,
        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,
        ex25,ex26,ex27,

        ex28,ex29,ex30,
        ex31,ex32,ex33,
        ex34,ex35,ex36,
        
        ex37,ex38,ex39
    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::3].copy()
        stepc2 = val1[1::3].copy()
        stepn1 = val1[2::3].copy()

        ex1,dex1 = s3rads3rodex1(stepn,stepc1,stepc2,stepn1,dt,params)
        ex2,dex2 = s3rads3rodex2(stepn,stepc1,stepc2,stepn1,dt,params)
        ex3,dex3 = s3rads3rodex3(stepn,stepc1,stepc2,stepn1,dt,params)
        ex4,dex4 = s3rads3rodex4(stepn,stepc1,stepc2,stepn1,dt,params)
        ex5,dex5 = s3rads3rodex5(stepn,stepc1,stepc2,stepn1,dt,params)
        ex6,dex6 = s3rads3rodex6(stepn,stepc1,stepc2,stepn1,dt,params)
        ex7,dex7 = s3rads3rodex7(stepn,stepc1,stepc2,stepn1,dt,params)
        ex8,dex8 = s3rads3rodex8(stepn,stepc1,stepc2,stepn1,dt,params)
        ex9,dex9 = s3rads3rodex9(stepn,stepc1,stepc2,stepn1,dt,params)

        ex10,dex10 = s3rads3rodex10(stepn,stepc1,stepc2,stepn1,dt,params)
        ex11,dex11 = s3rads3rodex11(stepn,stepc1,stepc2,stepn1,dt,params)
        ex12,dex12 = s3rads3rodex12(stepn,stepc1,stepc2,stepn1,dt,params)
        ex13,dex13 = s3rads3rodex13(stepn,stepc1,stepc2,stepn1,dt,params)
        ex14,dex14 = s3rads3rodex14(stepn,stepc1,stepc2,stepn1,dt,params)
        ex15,dex15 = s3rads3rodex15(stepn,stepc1,stepc2,stepn1,dt,params)
        ex16,dex16 = s3rads3rodex16(stepn,stepc1,stepc2,stepn1,dt,params)
        ex17,dex17 = s3rads3rodex17(stepn,stepc1,stepc2,stepn1,dt,params)
        ex18,dex18 = s3rads3rodex18(stepn,stepc1,stepc2,stepn1,dt,params)

        ex19,dex19 = s3rads3rodex19(stepn,stepc1,stepc2,stepn1,dt,params)
        ex20,dex20 = s3rads3rodex20(stepn,stepc1,stepc2,stepn1,dt,params)
        ex21,dex21 = s3rads3rodex21(stepn,stepc1,stepc2,stepn1,dt,params)
        ex22,dex22 = s3rads3rodex22(stepn,stepc1,stepc2,stepn1,dt,params)
        ex23,dex23 = s3rads3rodex23(stepn,stepc1,stepc2,stepn1,dt,params)
        ex24,dex24 = s3rads3rodex24(stepn,stepc1,stepc2,stepn1,dt,params)
        ex25,dex25 = s3rads3rodex25(stepn,stepc1,stepc2,stepn1,dt,params)
        ex26,dex26 = s3rads3rodex26(stepn,stepc1,stepc2,stepn1,dt,params)
        ex27,dex27 = s3rads3rodex27(stepn,stepc1,stepc2,stepn1,dt,params)

        ex28,dex28 = s3rads3rodex28(stepn,stepc1,stepc2,stepn1,dt,params)
        ex29,dex29 = s3rads3rodex29(stepn,stepc1,stepc2,stepn1,dt,params)
        ex30,dex30 = s3rads3rodex30(stepn,stepc1,stepc2,stepn1,dt,params)
        ex31,dex31 = s3rads3rodex31(stepn,stepc1,stepc2,stepn1,dt,params)
        ex32,dex32 = s3rads3rodex32(stepn,stepc1,stepc2,stepn1,dt,params)
        ex33,dex33 = s3rads3rodex33(stepn,stepc1,stepc2,stepn1,dt,params)
        ex34,dex34 = s3rads3rodex34(stepn,stepc1,stepc2,stepn1,dt,params)
        ex35,dex35 = s3rads3rodex35(stepn,stepc1,stepc2,stepn1,dt,params)
        ex36,dex36 = s3rads3rodex36(stepn,stepc1,stepc2,stepn1,dt,params)

        ex37,dex37 = s3rads3rodex37(stepn,stepc1,stepc2,stepn1,dt,params)
        ex38,dex38 = s3rads3rodex38(stepn,stepc1,stepc2,stepn1,dt,params)
        ex39,dex39 = s3rads3rodex39(stepn,stepc1,stepc2,stepn1,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,
            dex7,dex8,dex9,

            dex10,dex11,dex12,
            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,
            dex25,dex26,dex27,

            dex28,dex29,dex30,
            dex31,dex32,dex33,
            dex34,dex35,dex36,

            dex37,dex38,dex39

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,
            ex7,ex8,ex9,

            ex10,ex11,ex12,
            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,
            ex25,ex26,ex27,

            ex28,ex29,ex30,
            ex31,ex32,ex33,
            ex34,ex35,ex36,
            
            ex37,ex38,ex39
        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[2::3]



##########################
# Rigid Triangle Solvers #
##########################


####################################
# Two Stage Radau (Rigid DAE - H3) #
####################################

# This is the manual solver - it seems to work

def h3rads2tridae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],
        stepc1[1],stepc2[1],
        stepc1[2],stepc2[2],

        stepc1[3],stepc2[3],
        stepc1[4],stepc2[4],
        stepc1[5],stepc2[5],

        stepc1[6],stepc2[6],
        stepc1[7],stepc2[7],
        stepc1[8],stepc2[8],

        stepc1[9],stepc2[9],
        stepc1[10],stepc2[10],
        stepc1[11],stepc2[11],

        stepc1[12],stepc2[12],
        stepc1[13],stepc2[13],
        stepc1[14],stepc2[14],

        stepc1[15],stepc2[15],
        stepc1[16],stepc2[16],
        stepc1[17],stepc2[17],

        stepc1[18],stepc2[18],
        stepc1[19],stepc2[19],
        stepc1[20],stepc2[20]
        ])

    # p1
    ex1,dex1 = h3rads2triex1(stepn,stepc1,stepc2,dt,params)
    ex2,dex2 = h3rads2triex2(stepn,stepc1,stepc2,dt,params)
    ex3,dex3 = h3rads2triex3(stepn,stepc1,stepc2,dt,params)
    ex4,dex4 = h3rads2triex4(stepn,stepc1,stepc2,dt,params)
    ex5,dex5 = h3rads2triex5(stepn,stepc1,stepc2,dt,params)
    ex6,dex6 = h3rads2triex6(stepn,stepc1,stepc2,dt,params)

    # p2
    ex7,dex7 = h3rads2triex7(stepn,stepc1,stepc2,dt,params)
    ex8,dex8 = h3rads2triex8(stepn,stepc1,stepc2,dt,params)
    ex9,dex9 = h3rads2triex9(stepn,stepc1,stepc2,dt,params)
    ex10,dex10 = h3rads2triex10(stepn,stepc1,stepc2,dt,params)
    ex11,dex11 = h3rads2triex11(stepn,stepc1,stepc2,dt,params)
    ex12,dex12 = h3rads2triex12(stepn,stepc1,stepc2,dt,params)

    # p3
    ex13,dex13 = h3rads2triex13(stepn,stepc1,stepc2,dt,params)
    ex14,dex14 = h3rads2triex14(stepn,stepc1,stepc2,dt,params)
    ex15,dex15 = h3rads2triex15(stepn,stepc1,stepc2,dt,params)
    ex16,dex16 = h3rads2triex16(stepn,stepc1,stepc2,dt,params)
    ex17,dex17 = h3rads2triex17(stepn,stepc1,stepc2,dt,params)
    ex18,dex18 = h3rads2triex18(stepn,stepc1,stepc2,dt,params)

    #v1
    ex19,dex19 = h3rads2triex19(stepn,stepc1,stepc2,dt,params)
    ex20,dex20 = h3rads2triex20(stepn,stepc1,stepc2,dt,params)
    ex21,dex21 = h3rads2triex21(stepn,stepc1,stepc2,dt,params)
    ex22,dex22 = h3rads2triex22(stepn,stepc1,stepc2,dt,params)
    ex23,dex23 = h3rads2triex23(stepn,stepc1,stepc2,dt,params)
    ex24,dex24 = h3rads2triex24(stepn,stepc1,stepc2,dt,params)

    # v2
    ex25,dex25 = h3rads2triex25(stepn,stepc1,stepc2,dt,params)
    ex26,dex26 = h3rads2triex26(stepn,stepc1,stepc2,dt,params)
    ex27,dex27 = h3rads2triex27(stepn,stepc1,stepc2,dt,params)
    ex28,dex28 = h3rads2triex28(stepn,stepc1,stepc2,dt,params)
    ex29,dex29 = h3rads2triex29(stepn,stepc1,stepc2,dt,params)
    ex30,dex30 = h3rads2triex30(stepn,stepc1,stepc2,dt,params)

    # v3
    ex31,dex31 = h3rads2triex31(stepn,stepc1,stepc2,dt,params)
    ex32,dex32 = h3rads2triex32(stepn,stepc1,stepc2,dt,params)
    ex33,dex33 = h3rads2triex33(stepn,stepc1,stepc2,dt,params)
    ex34,dex34 = h3rads2triex34(stepn,stepc1,stepc2,dt,params)
    ex35,dex35 = h3rads2triex35(stepn,stepc1,stepc2,dt,params)
    ex36,dex36 = h3rads2triex36(stepn,stepc1,stepc2,dt,params)

    # lam
    ex37,dex37 = h3rads2triex37(stepn,stepc1,stepc2,dt,params)
    ex38,dex38 = h3rads2triex38(stepn,stepc1,stepc2,dt,params)
    ex39,dex39 = h3rads2triex39(stepn,stepc1,stepc2,dt,params)
    ex40,dex40 = h3rads2triex40(stepn,stepc1,stepc2,dt,params)
    ex41,dex41 = h3rads2triex41(stepn,stepc1,stepc2,dt,params)
    ex42,dex42 = h3rads2triex42(stepn,stepc1,stepc2,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,

        dex7,dex8,dex9,
        dex10,dex11,dex12,

        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,

        dex25,dex26,dex27,
        dex28,dex29,dex30,

        dex31,dex32,dex33,
        dex34,dex35,dex36,

        dex37,dex38,dex39,
        dex40,dex41,dex42

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,

        ex7,ex8,ex9,
        ex10,ex11,ex12,

        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,

        ex25,ex26,ex27,
        ex28,ex29,ex30,

        ex31,ex32,ex33,
        ex34,ex35,ex36,

        ex37,ex38,ex39,
        ex40,ex41,ex42
    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::2].copy()
        stepc2 = val1[1::2].copy()

        # p1
        ex1,dex1 = h3rads2triex1(stepn,stepc1,stepc2,dt,params)
        ex2,dex2 = h3rads2triex2(stepn,stepc1,stepc2,dt,params)
        ex3,dex3 = h3rads2triex3(stepn,stepc1,stepc2,dt,params)
        ex4,dex4 = h3rads2triex4(stepn,stepc1,stepc2,dt,params)
        ex5,dex5 = h3rads2triex5(stepn,stepc1,stepc2,dt,params)
        ex6,dex6 = h3rads2triex6(stepn,stepc1,stepc2,dt,params)

        # p2
        ex7,dex7 = h3rads2triex7(stepn,stepc1,stepc2,dt,params)
        ex8,dex8 = h3rads2triex8(stepn,stepc1,stepc2,dt,params)
        ex9,dex9 = h3rads2triex9(stepn,stepc1,stepc2,dt,params)
        ex10,dex10 = h3rads2triex10(stepn,stepc1,stepc2,dt,params)
        ex11,dex11 = h3rads2triex11(stepn,stepc1,stepc2,dt,params)
        ex12,dex12 = h3rads2triex12(stepn,stepc1,stepc2,dt,params)

        # p3
        ex13,dex13 = h3rads2triex13(stepn,stepc1,stepc2,dt,params)
        ex14,dex14 = h3rads2triex14(stepn,stepc1,stepc2,dt,params)
        ex15,dex15 = h3rads2triex15(stepn,stepc1,stepc2,dt,params)
        ex16,dex16 = h3rads2triex16(stepn,stepc1,stepc2,dt,params)
        ex17,dex17 = h3rads2triex17(stepn,stepc1,stepc2,dt,params)
        ex18,dex18 = h3rads2triex18(stepn,stepc1,stepc2,dt,params)

        #v1
        ex19,dex19 = h3rads2triex19(stepn,stepc1,stepc2,dt,params)
        ex20,dex20 = h3rads2triex20(stepn,stepc1,stepc2,dt,params)
        ex21,dex21 = h3rads2triex21(stepn,stepc1,stepc2,dt,params)
        ex22,dex22 = h3rads2triex22(stepn,stepc1,stepc2,dt,params)
        ex23,dex23 = h3rads2triex23(stepn,stepc1,stepc2,dt,params)
        ex24,dex24 = h3rads2triex24(stepn,stepc1,stepc2,dt,params)

        # v2
        ex25,dex25 = h3rads2triex25(stepn,stepc1,stepc2,dt,params)
        ex26,dex26 = h3rads2triex26(stepn,stepc1,stepc2,dt,params)
        ex27,dex27 = h3rads2triex27(stepn,stepc1,stepc2,dt,params)
        ex28,dex28 = h3rads2triex28(stepn,stepc1,stepc2,dt,params)
        ex29,dex29 = h3rads2triex29(stepn,stepc1,stepc2,dt,params)
        ex30,dex30 = h3rads2triex30(stepn,stepc1,stepc2,dt,params)

        # v3
        ex31,dex31 = h3rads2triex31(stepn,stepc1,stepc2,dt,params)
        ex32,dex32 = h3rads2triex32(stepn,stepc1,stepc2,dt,params)
        ex33,dex33 = h3rads2triex33(stepn,stepc1,stepc2,dt,params)
        ex34,dex34 = h3rads2triex34(stepn,stepc1,stepc2,dt,params)
        ex35,dex35 = h3rads2triex35(stepn,stepc1,stepc2,dt,params)
        ex36,dex36 = h3rads2triex36(stepn,stepc1,stepc2,dt,params)

        # lam
        ex37,dex37 = h3rads2triex37(stepn,stepc1,stepc2,dt,params)
        ex38,dex38 = h3rads2triex38(stepn,stepc1,stepc2,dt,params)
        ex39,dex39 = h3rads2triex39(stepn,stepc1,stepc2,dt,params)
        ex40,dex40 = h3rads2triex40(stepn,stepc1,stepc2,dt,params)
        ex41,dex41 = h3rads2triex41(stepn,stepc1,stepc2,dt,params)
        ex42,dex42 = h3rads2triex42(stepn,stepc1,stepc2,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,

            dex7,dex8,dex9,
            dex10,dex11,dex12,

            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,

            dex25,dex26,dex27,
            dex28,dex29,dex30,

            dex31,dex32,dex33,
            dex34,dex35,dex36,

            dex37,dex38,dex39,
            dex40,dex41,dex42

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,

            ex7,ex8,ex9,
            ex10,ex11,ex12,

            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,

            ex25,ex26,ex27,
            ex28,ex29,ex30,

            ex31,ex32,ex33,
            ex34,ex35,ex36,

            ex37,ex38,ex39,
            ex40,ex41,ex42
        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[1::2]

######################################
# Three Stage Radau (Rigid DAE - H3) #
######################################

# This is the manual solver - it seems to work

def h3rads3tridae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    stepn1 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],stepn1[0],
        stepc1[1],stepc2[1],stepn1[1],
        stepc1[2],stepc2[2],stepn1[2],

        stepc1[3],stepc2[3],stepn1[3],
        stepc1[4],stepc2[4],stepn1[4],
        stepc1[5],stepc2[5],stepn1[5],

        stepc1[6],stepc2[6],stepn1[6],
        stepc1[7],stepc2[7],stepn1[7],
        stepc1[8],stepc2[8],stepn1[8],

        stepc1[9],stepc2[9],stepn1[9],
        stepc1[10],stepc2[10],stepn1[10],
        stepc1[11],stepc2[11],stepn1[11],

        stepc1[12],stepc2[12],stepn1[12],
        stepc1[13],stepc2[13],stepn1[13],
        stepc1[14],stepc2[14],stepn1[14],

        stepc1[15],stepc2[15],stepn1[15],
        stepc1[16],stepc2[16],stepn1[16],
        stepc1[17],stepc2[17],stepn1[17],

        stepc1[18],stepc2[18],stepn1[18],
        stepc1[19],stepc2[19],stepn1[19],
        stepc1[20],stepc2[20],stepn1[20]
        ])

    # p1
    ex1,dex1 = h3rads3triex1(stepn,stepc1,stepc2,stepn1,dt,params)
    ex2,dex2 = h3rads3triex2(stepn,stepc1,stepc2,stepn1,dt,params)
    ex3,dex3 = h3rads3triex3(stepn,stepc1,stepc2,stepn1,dt,params)
    ex4,dex4 = h3rads3triex4(stepn,stepc1,stepc2,stepn1,dt,params)
    ex5,dex5 = h3rads3triex5(stepn,stepc1,stepc2,stepn1,dt,params)
    ex6,dex6 = h3rads3triex6(stepn,stepc1,stepc2,stepn1,dt,params)
    ex7,dex7 = h3rads3triex7(stepn,stepc1,stepc2,stepn1,dt,params)
    ex8,dex8 = h3rads3triex8(stepn,stepc1,stepc2,stepn1,dt,params)
    ex9,dex9 = h3rads3triex9(stepn,stepc1,stepc2,stepn1,dt,params)

    # p2
    ex10,dex10 = h3rads3triex10(stepn,stepc1,stepc2,stepn1,dt,params)
    ex11,dex11 = h3rads3triex11(stepn,stepc1,stepc2,stepn1,dt,params)
    ex12,dex12 = h3rads3triex12(stepn,stepc1,stepc2,stepn1,dt,params)
    ex13,dex13 = h3rads3triex13(stepn,stepc1,stepc2,stepn1,dt,params)
    ex14,dex14 = h3rads3triex14(stepn,stepc1,stepc2,stepn1,dt,params)
    ex15,dex15 = h3rads3triex15(stepn,stepc1,stepc2,stepn1,dt,params)
    ex16,dex16 = h3rads3triex16(stepn,stepc1,stepc2,stepn1,dt,params)
    ex17,dex17 = h3rads3triex17(stepn,stepc1,stepc2,stepn1,dt,params)
    ex18,dex18 = h3rads3triex18(stepn,stepc1,stepc2,stepn1,dt,params)

    #p3
    ex19,dex19 = h3rads3triex19(stepn,stepc1,stepc2,stepn1,dt,params)
    ex20,dex20 = h3rads3triex20(stepn,stepc1,stepc2,stepn1,dt,params)
    ex21,dex21 = h3rads3triex21(stepn,stepc1,stepc2,stepn1,dt,params)
    ex22,dex22 = h3rads3triex22(stepn,stepc1,stepc2,stepn1,dt,params)
    ex23,dex23 = h3rads3triex23(stepn,stepc1,stepc2,stepn1,dt,params)
    ex24,dex24 = h3rads3triex24(stepn,stepc1,stepc2,stepn1,dt,params)
    ex25,dex25 = h3rads3triex25(stepn,stepc1,stepc2,stepn1,dt,params)
    ex26,dex26 = h3rads3triex26(stepn,stepc1,stepc2,stepn1,dt,params)
    ex27,dex27 = h3rads3triex27(stepn,stepc1,stepc2,stepn1,dt,params)

    # v1
    ex28,dex28 = h3rads3triex28(stepn,stepc1,stepc2,stepn1,dt,params)
    ex29,dex29 = h3rads3triex29(stepn,stepc1,stepc2,stepn1,dt,params)
    ex30,dex30 = h3rads3triex30(stepn,stepc1,stepc2,stepn1,dt,params)
    ex31,dex31 = h3rads3triex31(stepn,stepc1,stepc2,stepn1,dt,params)
    ex32,dex32 = h3rads3triex32(stepn,stepc1,stepc2,stepn1,dt,params)
    ex33,dex33 = h3rads3triex33(stepn,stepc1,stepc2,stepn1,dt,params)
    ex34,dex34 = h3rads3triex34(stepn,stepc1,stepc2,stepn1,dt,params)
    ex35,dex35 = h3rads3triex35(stepn,stepc1,stepc2,stepn1,dt,params)
    ex36,dex36 = h3rads3triex36(stepn,stepc1,stepc2,stepn1,dt,params)

    # v2
    ex37,dex37 = h3rads3triex37(stepn,stepc1,stepc2,stepn1,dt,params)
    ex38,dex38 = h3rads3triex38(stepn,stepc1,stepc2,stepn1,dt,params)
    ex39,dex39 = h3rads3triex39(stepn,stepc1,stepc2,stepn1,dt,params)
    ex40,dex40 = h3rads3triex40(stepn,stepc1,stepc2,stepn1,dt,params)
    ex41,dex41 = h3rads3triex41(stepn,stepc1,stepc2,stepn1,dt,params)
    ex42,dex42 = h3rads3triex42(stepn,stepc1,stepc2,stepn1,dt,params)
    ex43,dex43 = h3rads3triex43(stepn,stepc1,stepc2,stepn1,dt,params)
    ex44,dex44 = h3rads3triex44(stepn,stepc1,stepc2,stepn1,dt,params)
    ex45,dex45 = h3rads3triex45(stepn,stepc1,stepc2,stepn1,dt,params)

    # v3
    ex46,dex46 = h3rads3triex46(stepn,stepc1,stepc2,stepn1,dt,params)
    ex47,dex47 = h3rads3triex47(stepn,stepc1,stepc2,stepn1,dt,params)
    ex48,dex48 = h3rads3triex48(stepn,stepc1,stepc2,stepn1,dt,params)
    ex49,dex49 = h3rads3triex49(stepn,stepc1,stepc2,stepn1,dt,params)
    ex50,dex50 = h3rads3triex50(stepn,stepc1,stepc2,stepn1,dt,params)
    ex51,dex51 = h3rads3triex51(stepn,stepc1,stepc2,stepn1,dt,params)
    ex52,dex52 = h3rads3triex52(stepn,stepc1,stepc2,stepn1,dt,params)
    ex53,dex53 = h3rads3triex53(stepn,stepc1,stepc2,stepn1,dt,params)
    ex54,dex54 = h3rads3triex54(stepn,stepc1,stepc2,stepn1,dt,params)

    # lam
    ex55,dex55 = h3rads3triex55(stepn,stepc1,stepc2,stepn1,dt,params)
    ex56,dex56 = h3rads3triex56(stepn,stepc1,stepc2,stepn1,dt,params)
    ex57,dex57 = h3rads3triex57(stepn,stepc1,stepc2,stepn1,dt,params)
    ex58,dex58 = h3rads3triex58(stepn,stepc1,stepc2,stepn1,dt,params)
    ex59,dex59 = h3rads3triex59(stepn,stepc1,stepc2,stepn1,dt,params)
    ex60,dex60 = h3rads3triex60(stepn,stepc1,stepc2,stepn1,dt,params)
    ex61,dex61 = h3rads3triex61(stepn,stepc1,stepc2,stepn1,dt,params)
    ex62,dex62 = h3rads3triex62(stepn,stepc1,stepc2,stepn1,dt,params)
    ex63,dex63 = h3rads3triex63(stepn,stepc1,stepc2,stepn1,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,
        dex7,dex8,dex9,

        dex10,dex11,dex12,
        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,
        dex25,dex26,dex27,

        dex28,dex29,dex30,
        dex31,dex32,dex33,
        dex34,dex35,dex36,

        dex37,dex38,dex39,
        dex40,dex41,dex42,
        dex43,dex44,dex45,

        dex46,dex47,dex48,
        dex49,dex50,dex51,
        dex52,dex53,dex54,

        dex55,dex56,dex57,
        dex58,dex59,dex60,
        dex61,dex62,dex63

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,
        ex7,ex8,ex9,

        ex10,ex11,ex12,
        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,
        ex25,ex26,ex27,

        ex28,ex29,ex30,
        ex31,ex32,ex33,
        ex34,ex35,ex36,

        ex37,ex38,ex39,
        ex40,ex41,ex42,
        ex43,ex44,ex45,

        ex46,ex47,ex48,
        ex49,ex50,ex51,
        ex52,ex53,ex54,

        ex55,ex56,ex57,
        ex58,ex59,ex60,
        ex61,ex62,ex63

    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::3].copy()
        stepc2 = val1[1::3].copy()
        stepn1 = val1[2::3].copy()

        # p1
        ex1,dex1 = h3rads3triex1(stepn,stepc1,stepc2,stepn1,dt,params)
        ex2,dex2 = h3rads3triex2(stepn,stepc1,stepc2,stepn1,dt,params)
        ex3,dex3 = h3rads3triex3(stepn,stepc1,stepc2,stepn1,dt,params)
        ex4,dex4 = h3rads3triex4(stepn,stepc1,stepc2,stepn1,dt,params)
        ex5,dex5 = h3rads3triex5(stepn,stepc1,stepc2,stepn1,dt,params)
        ex6,dex6 = h3rads3triex6(stepn,stepc1,stepc2,stepn1,dt,params)
        ex7,dex7 = h3rads3triex7(stepn,stepc1,stepc2,stepn1,dt,params)
        ex8,dex8 = h3rads3triex8(stepn,stepc1,stepc2,stepn1,dt,params)
        ex9,dex9 = h3rads3triex9(stepn,stepc1,stepc2,stepn1,dt,params)

        # p2
        ex10,dex10 = h3rads3triex10(stepn,stepc1,stepc2,stepn1,dt,params)
        ex11,dex11 = h3rads3triex11(stepn,stepc1,stepc2,stepn1,dt,params)
        ex12,dex12 = h3rads3triex12(stepn,stepc1,stepc2,stepn1,dt,params)
        ex13,dex13 = h3rads3triex13(stepn,stepc1,stepc2,stepn1,dt,params)
        ex14,dex14 = h3rads3triex14(stepn,stepc1,stepc2,stepn1,dt,params)
        ex15,dex15 = h3rads3triex15(stepn,stepc1,stepc2,stepn1,dt,params)
        ex16,dex16 = h3rads3triex16(stepn,stepc1,stepc2,stepn1,dt,params)
        ex17,dex17 = h3rads3triex17(stepn,stepc1,stepc2,stepn1,dt,params)
        ex18,dex18 = h3rads3triex18(stepn,stepc1,stepc2,stepn1,dt,params)

        #p3
        ex19,dex19 = h3rads3triex19(stepn,stepc1,stepc2,stepn1,dt,params)
        ex20,dex20 = h3rads3triex20(stepn,stepc1,stepc2,stepn1,dt,params)
        ex21,dex21 = h3rads3triex21(stepn,stepc1,stepc2,stepn1,dt,params)
        ex22,dex22 = h3rads3triex22(stepn,stepc1,stepc2,stepn1,dt,params)
        ex23,dex23 = h3rads3triex23(stepn,stepc1,stepc2,stepn1,dt,params)
        ex24,dex24 = h3rads3triex24(stepn,stepc1,stepc2,stepn1,dt,params)
        ex25,dex25 = h3rads3triex25(stepn,stepc1,stepc2,stepn1,dt,params)
        ex26,dex26 = h3rads3triex26(stepn,stepc1,stepc2,stepn1,dt,params)
        ex27,dex27 = h3rads3triex27(stepn,stepc1,stepc2,stepn1,dt,params)

        # v1
        ex28,dex28 = h3rads3triex28(stepn,stepc1,stepc2,stepn1,dt,params)
        ex29,dex29 = h3rads3triex29(stepn,stepc1,stepc2,stepn1,dt,params)
        ex30,dex30 = h3rads3triex30(stepn,stepc1,stepc2,stepn1,dt,params)
        ex31,dex31 = h3rads3triex31(stepn,stepc1,stepc2,stepn1,dt,params)
        ex32,dex32 = h3rads3triex32(stepn,stepc1,stepc2,stepn1,dt,params)
        ex33,dex33 = h3rads3triex33(stepn,stepc1,stepc2,stepn1,dt,params)
        ex34,dex34 = h3rads3triex34(stepn,stepc1,stepc2,stepn1,dt,params)
        ex35,dex35 = h3rads3triex35(stepn,stepc1,stepc2,stepn1,dt,params)
        ex36,dex36 = h3rads3triex36(stepn,stepc1,stepc2,stepn1,dt,params)

        # v2
        ex37,dex37 = h3rads3triex37(stepn,stepc1,stepc2,stepn1,dt,params)
        ex38,dex38 = h3rads3triex38(stepn,stepc1,stepc2,stepn1,dt,params)
        ex39,dex39 = h3rads3triex39(stepn,stepc1,stepc2,stepn1,dt,params)
        ex40,dex40 = h3rads3triex40(stepn,stepc1,stepc2,stepn1,dt,params)
        ex41,dex41 = h3rads3triex41(stepn,stepc1,stepc2,stepn1,dt,params)
        ex42,dex42 = h3rads3triex42(stepn,stepc1,stepc2,stepn1,dt,params)
        ex43,dex43 = h3rads3triex43(stepn,stepc1,stepc2,stepn1,dt,params)
        ex44,dex44 = h3rads3triex44(stepn,stepc1,stepc2,stepn1,dt,params)
        ex45,dex45 = h3rads3triex45(stepn,stepc1,stepc2,stepn1,dt,params)

        # v3
        ex46,dex46 = h3rads3triex46(stepn,stepc1,stepc2,stepn1,dt,params)
        ex47,dex47 = h3rads3triex47(stepn,stepc1,stepc2,stepn1,dt,params)
        ex48,dex48 = h3rads3triex48(stepn,stepc1,stepc2,stepn1,dt,params)
        ex49,dex49 = h3rads3triex49(stepn,stepc1,stepc2,stepn1,dt,params)
        ex50,dex50 = h3rads3triex50(stepn,stepc1,stepc2,stepn1,dt,params)
        ex51,dex51 = h3rads3triex51(stepn,stepc1,stepc2,stepn1,dt,params)
        ex52,dex52 = h3rads3triex52(stepn,stepc1,stepc2,stepn1,dt,params)
        ex53,dex53 = h3rads3triex53(stepn,stepc1,stepc2,stepn1,dt,params)
        ex54,dex54 = h3rads3triex54(stepn,stepc1,stepc2,stepn1,dt,params)

        # lam
        ex55,dex55 = h3rads3triex55(stepn,stepc1,stepc2,stepn1,dt,params)
        ex56,dex56 = h3rads3triex56(stepn,stepc1,stepc2,stepn1,dt,params)
        ex57,dex57 = h3rads3triex57(stepn,stepc1,stepc2,stepn1,dt,params)
        ex58,dex58 = h3rads3triex58(stepn,stepc1,stepc2,stepn1,dt,params)
        ex59,dex59 = h3rads3triex59(stepn,stepc1,stepc2,stepn1,dt,params)
        ex60,dex60 = h3rads3triex60(stepn,stepc1,stepc2,stepn1,dt,params)
        ex61,dex61 = h3rads3triex61(stepn,stepc1,stepc2,stepn1,dt,params)
        ex62,dex62 = h3rads3triex62(stepn,stepc1,stepc2,stepn1,dt,params)
        ex63,dex63 = h3rads3triex63(stepn,stepc1,stepc2,stepn1,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,
            dex7,dex8,dex9,

            dex10,dex11,dex12,
            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,
            dex25,dex26,dex27,

            dex28,dex29,dex30,
            dex31,dex32,dex33,
            dex34,dex35,dex36,

            dex37,dex38,dex39,
            dex40,dex41,dex42,
            dex43,dex44,dex45,

            dex46,dex47,dex48,
            dex49,dex50,dex51,
            dex52,dex53,dex54,

            dex55,dex56,dex57,
            dex58,dex59,dex60,
            dex61,dex62,dex63

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,
            ex7,ex8,ex9,

            ex10,ex11,ex12,
            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,
            ex25,ex26,ex27,

            ex28,ex29,ex30,
            ex31,ex32,ex33,
            ex34,ex35,ex36,

            ex37,ex38,ex39,
            ex40,ex41,ex42,
            ex43,ex44,ex45,

            ex46,ex47,ex48,
            ex49,ex50,ex51,
            ex52,ex53,ex54,

            ex55,ex56,ex57,
            ex58,ex59,ex60,
            ex61,ex62,ex63

        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[2::3]

####################################
# Two Stage Radau (Rigid DAE - S3) #
####################################

# This is the manual solver - it seems to work

def s3rads2tridae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],
        stepc1[1],stepc2[1],
        stepc1[2],stepc2[2],

        stepc1[3],stepc2[3],
        stepc1[4],stepc2[4],
        stepc1[5],stepc2[5],

        stepc1[6],stepc2[6],
        stepc1[7],stepc2[7],
        stepc1[8],stepc2[8],

        stepc1[9],stepc2[9],
        stepc1[10],stepc2[10],
        stepc1[11],stepc2[11],

        stepc1[12],stepc2[12],
        stepc1[13],stepc2[13],
        stepc1[14],stepc2[14],

        stepc1[15],stepc2[15],
        stepc1[16],stepc2[16],
        stepc1[17],stepc2[17],

        stepc1[18],stepc2[18],
        stepc1[19],stepc2[19],
        stepc1[20],stepc2[20]
        ])

    # p1
    ex1,dex1 = s3rads2triex1(stepn,stepc1,stepc2,dt,params)
    ex2,dex2 = s3rads2triex2(stepn,stepc1,stepc2,dt,params)
    ex3,dex3 = s3rads2triex3(stepn,stepc1,stepc2,dt,params)
    ex4,dex4 = s3rads2triex4(stepn,stepc1,stepc2,dt,params)
    ex5,dex5 = s3rads2triex5(stepn,stepc1,stepc2,dt,params)
    ex6,dex6 = s3rads2triex6(stepn,stepc1,stepc2,dt,params)

    # p2
    ex7,dex7   = s3rads2triex7(stepn,stepc1,stepc2,dt,params)
    ex8,dex8   = s3rads2triex8(stepn,stepc1,stepc2,dt,params)
    ex9,dex9   = s3rads2triex9(stepn,stepc1,stepc2,dt,params)
    ex10,dex10 = s3rads2triex10(stepn,stepc1,stepc2,dt,params)
    ex11,dex11 = s3rads2triex11(stepn,stepc1,stepc2,dt,params)
    ex12,dex12 = s3rads2triex12(stepn,stepc1,stepc2,dt,params)

    # p3
    ex13,dex13 = s3rads2triex13(stepn,stepc1,stepc2,dt,params)
    ex14,dex14 = s3rads2triex14(stepn,stepc1,stepc2,dt,params)
    ex15,dex15 = s3rads2triex15(stepn,stepc1,stepc2,dt,params)
    ex16,dex16 = s3rads2triex16(stepn,stepc1,stepc2,dt,params)
    ex17,dex17 = s3rads2triex17(stepn,stepc1,stepc2,dt,params)
    ex18,dex18 = s3rads2triex18(stepn,stepc1,stepc2,dt,params)

    #v1
    ex19,dex19 = s3rads2triex19(stepn,stepc1,stepc2,dt,params)
    ex20,dex20 = s3rads2triex20(stepn,stepc1,stepc2,dt,params)
    ex21,dex21 = s3rads2triex21(stepn,stepc1,stepc2,dt,params)
    ex22,dex22 = s3rads2triex22(stepn,stepc1,stepc2,dt,params)
    ex23,dex23 = s3rads2triex23(stepn,stepc1,stepc2,dt,params)
    ex24,dex24 = s3rads2triex24(stepn,stepc1,stepc2,dt,params)

    # v2
    ex25,dex25 = s3rads2triex25(stepn,stepc1,stepc2,dt,params)
    ex26,dex26 = s3rads2triex26(stepn,stepc1,stepc2,dt,params)
    ex27,dex27 = s3rads2triex27(stepn,stepc1,stepc2,dt,params)
    ex28,dex28 = s3rads2triex28(stepn,stepc1,stepc2,dt,params)
    ex29,dex29 = s3rads2triex29(stepn,stepc1,stepc2,dt,params)
    ex30,dex30 = s3rads2triex30(stepn,stepc1,stepc2,dt,params)

    # v3
    ex31,dex31 = s3rads2triex31(stepn,stepc1,stepc2,dt,params)
    ex32,dex32 = s3rads2triex32(stepn,stepc1,stepc2,dt,params)
    ex33,dex33 = s3rads2triex33(stepn,stepc1,stepc2,dt,params)
    ex34,dex34 = s3rads2triex34(stepn,stepc1,stepc2,dt,params)
    ex35,dex35 = s3rads2triex35(stepn,stepc1,stepc2,dt,params)
    ex36,dex36 = s3rads2triex36(stepn,stepc1,stepc2,dt,params)

    # lam
    ex37,dex37 = s3rads2triex37(stepn,stepc1,stepc2,dt,params)
    ex38,dex38 = s3rads2triex38(stepn,stepc1,stepc2,dt,params)
    ex39,dex39 = s3rads2triex39(stepn,stepc1,stepc2,dt,params)
    ex40,dex40 = s3rads2triex40(stepn,stepc1,stepc2,dt,params)
    ex41,dex41 = s3rads2triex41(stepn,stepc1,stepc2,dt,params)
    ex42,dex42 = s3rads2triex42(stepn,stepc1,stepc2,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,

        dex7,dex8,dex9,
        dex10,dex11,dex12,

        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,

        dex25,dex26,dex27,
        dex28,dex29,dex30,

        dex31,dex32,dex33,
        dex34,dex35,dex36,

        dex37,dex38,dex39,
        dex40,dex41,dex42

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,

        ex7,ex8,ex9,
        ex10,ex11,ex12,

        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,

        ex25,ex26,ex27,
        ex28,ex29,ex30,

        ex31,ex32,ex33,
        ex34,ex35,ex36,

        ex37,ex38,ex39,
        ex40,ex41,ex42
    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::2].copy()
        stepc2 = val1[1::2].copy()

        # p1
        ex1,dex1 = s3rads2triex1(stepn,stepc1,stepc2,dt,params)
        ex2,dex2 = s3rads2triex2(stepn,stepc1,stepc2,dt,params)
        ex3,dex3 = s3rads2triex3(stepn,stepc1,stepc2,dt,params)
        ex4,dex4 = s3rads2triex4(stepn,stepc1,stepc2,dt,params)
        ex5,dex5 = s3rads2triex5(stepn,stepc1,stepc2,dt,params)
        ex6,dex6 = s3rads2triex6(stepn,stepc1,stepc2,dt,params)

        # p2
        ex7,dex7   = s3rads2triex7(stepn,stepc1,stepc2,dt,params)
        ex8,dex8   = s3rads2triex8(stepn,stepc1,stepc2,dt,params)
        ex9,dex9   = s3rads2triex9(stepn,stepc1,stepc2,dt,params)
        ex10,dex10 = s3rads2triex10(stepn,stepc1,stepc2,dt,params)
        ex11,dex11 = s3rads2triex11(stepn,stepc1,stepc2,dt,params)
        ex12,dex12 = s3rads2triex12(stepn,stepc1,stepc2,dt,params)

        # p3
        ex13,dex13 = s3rads2triex13(stepn,stepc1,stepc2,dt,params)
        ex14,dex14 = s3rads2triex14(stepn,stepc1,stepc2,dt,params)
        ex15,dex15 = s3rads2triex15(stepn,stepc1,stepc2,dt,params)
        ex16,dex16 = s3rads2triex16(stepn,stepc1,stepc2,dt,params)
        ex17,dex17 = s3rads2triex17(stepn,stepc1,stepc2,dt,params)
        ex18,dex18 = s3rads2triex18(stepn,stepc1,stepc2,dt,params)

        #v1
        ex19,dex19 = s3rads2triex19(stepn,stepc1,stepc2,dt,params)
        ex20,dex20 = s3rads2triex20(stepn,stepc1,stepc2,dt,params)
        ex21,dex21 = s3rads2triex21(stepn,stepc1,stepc2,dt,params)
        ex22,dex22 = s3rads2triex22(stepn,stepc1,stepc2,dt,params)
        ex23,dex23 = s3rads2triex23(stepn,stepc1,stepc2,dt,params)
        ex24,dex24 = s3rads2triex24(stepn,stepc1,stepc2,dt,params)

        # v2
        ex25,dex25 = s3rads2triex25(stepn,stepc1,stepc2,dt,params)
        ex26,dex26 = s3rads2triex26(stepn,stepc1,stepc2,dt,params)
        ex27,dex27 = s3rads2triex27(stepn,stepc1,stepc2,dt,params)
        ex28,dex28 = s3rads2triex28(stepn,stepc1,stepc2,dt,params)
        ex29,dex29 = s3rads2triex29(stepn,stepc1,stepc2,dt,params)
        ex30,dex30 = s3rads2triex30(stepn,stepc1,stepc2,dt,params)

        # v3
        ex31,dex31 = s3rads2triex31(stepn,stepc1,stepc2,dt,params)
        ex32,dex32 = s3rads2triex32(stepn,stepc1,stepc2,dt,params)
        ex33,dex33 = s3rads2triex33(stepn,stepc1,stepc2,dt,params)
        ex34,dex34 = s3rads2triex34(stepn,stepc1,stepc2,dt,params)
        ex35,dex35 = s3rads2triex35(stepn,stepc1,stepc2,dt,params)
        ex36,dex36 = s3rads2triex36(stepn,stepc1,stepc2,dt,params)

        # lam
        ex37,dex37 = s3rads2triex37(stepn,stepc1,stepc2,dt,params)
        ex38,dex38 = s3rads2triex38(stepn,stepc1,stepc2,dt,params)
        ex39,dex39 = s3rads2triex39(stepn,stepc1,stepc2,dt,params)
        ex40,dex40 = s3rads2triex40(stepn,stepc1,stepc2,dt,params)
        ex41,dex41 = s3rads2triex41(stepn,stepc1,stepc2,dt,params)
        ex42,dex42 = s3rads2triex42(stepn,stepc1,stepc2,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,

            dex7,dex8,dex9,
            dex10,dex11,dex12,

            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,

            dex25,dex26,dex27,
            dex28,dex29,dex30,

            dex31,dex32,dex33,
            dex34,dex35,dex36,

            dex37,dex38,dex39,
            dex40,dex41,dex42

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,

            ex7,ex8,ex9,
            ex10,ex11,ex12,

            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,

            ex25,ex26,ex27,
            ex28,ex29,ex30,

            ex31,ex32,ex33,
            ex34,ex35,ex36,

            ex37,ex38,ex39,
            ex40,ex41,ex42
        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[1::2]

######################################
# Three Stage Radau (Rigid DAE - S3) #
######################################

# This is the manual solver - it seems to work

def s3rads3tridae(startvec, params, dt, tol = 1e-10, imax = 100):
    stepn = startvec.copy()
    stepc1 = startvec.copy()
    stepc2 = startvec.copy()
    stepn1 = startvec.copy()
    initvec = np.array([
        stepc1[0],stepc2[0],stepn1[0],
        stepc1[1],stepc2[1],stepn1[1],
        stepc1[2],stepc2[2],stepn1[2],

        stepc1[3],stepc2[3],stepn1[3],
        stepc1[4],stepc2[4],stepn1[4],
        stepc1[5],stepc2[5],stepn1[5],

        stepc1[6],stepc2[6],stepn1[6],
        stepc1[7],stepc2[7],stepn1[7],
        stepc1[8],stepc2[8],stepn1[8],

        stepc1[9],stepc2[9],stepn1[9],
        stepc1[10],stepc2[10],stepn1[10],
        stepc1[11],stepc2[11],stepn1[11],

        stepc1[12],stepc2[12],stepn1[12],
        stepc1[13],stepc2[13],stepn1[13],
        stepc1[14],stepc2[14],stepn1[14],

        stepc1[15],stepc2[15],stepn1[15],
        stepc1[16],stepc2[16],stepn1[16],
        stepc1[17],stepc2[17],stepn1[17],

        stepc1[18],stepc2[18],stepn1[18],
        stepc1[19],stepc2[19],stepn1[19],
        stepc1[20],stepc2[20],stepn1[20]
        ])

    # p1
    ex1,dex1 = s3rads3triex1(stepn,stepc1,stepc2,stepn1,dt,params)
    ex2,dex2 = s3rads3triex2(stepn,stepc1,stepc2,stepn1,dt,params)
    ex3,dex3 = s3rads3triex3(stepn,stepc1,stepc2,stepn1,dt,params)
    ex4,dex4 = s3rads3triex4(stepn,stepc1,stepc2,stepn1,dt,params)
    ex5,dex5 = s3rads3triex5(stepn,stepc1,stepc2,stepn1,dt,params)
    ex6,dex6 = s3rads3triex6(stepn,stepc1,stepc2,stepn1,dt,params)
    ex7,dex7 = s3rads3triex7(stepn,stepc1,stepc2,stepn1,dt,params)
    ex8,dex8 = s3rads3triex8(stepn,stepc1,stepc2,stepn1,dt,params)
    ex9,dex9 = s3rads3triex9(stepn,stepc1,stepc2,stepn1,dt,params)

    # p2
    ex10,dex10 = s3rads3triex10(stepn,stepc1,stepc2,stepn1,dt,params)
    ex11,dex11 = s3rads3triex11(stepn,stepc1,stepc2,stepn1,dt,params)
    ex12,dex12 = s3rads3triex12(stepn,stepc1,stepc2,stepn1,dt,params)
    ex13,dex13 = s3rads3triex13(stepn,stepc1,stepc2,stepn1,dt,params)
    ex14,dex14 = s3rads3triex14(stepn,stepc1,stepc2,stepn1,dt,params)
    ex15,dex15 = s3rads3triex15(stepn,stepc1,stepc2,stepn1,dt,params)
    ex16,dex16 = s3rads3triex16(stepn,stepc1,stepc2,stepn1,dt,params)
    ex17,dex17 = s3rads3triex17(stepn,stepc1,stepc2,stepn1,dt,params)
    ex18,dex18 = s3rads3triex18(stepn,stepc1,stepc2,stepn1,dt,params)

    #p3
    ex19,dex19 = s3rads3triex19(stepn,stepc1,stepc2,stepn1,dt,params)
    ex20,dex20 = s3rads3triex20(stepn,stepc1,stepc2,stepn1,dt,params)
    ex21,dex21 = s3rads3triex21(stepn,stepc1,stepc2,stepn1,dt,params)
    ex22,dex22 = s3rads3triex22(stepn,stepc1,stepc2,stepn1,dt,params)
    ex23,dex23 = s3rads3triex23(stepn,stepc1,stepc2,stepn1,dt,params)
    ex24,dex24 = s3rads3triex24(stepn,stepc1,stepc2,stepn1,dt,params)
    ex25,dex25 = s3rads3triex25(stepn,stepc1,stepc2,stepn1,dt,params)
    ex26,dex26 = s3rads3triex26(stepn,stepc1,stepc2,stepn1,dt,params)
    ex27,dex27 = s3rads3triex27(stepn,stepc1,stepc2,stepn1,dt,params)

    # v1
    ex28,dex28 = s3rads3triex28(stepn,stepc1,stepc2,stepn1,dt,params)
    ex29,dex29 = s3rads3triex29(stepn,stepc1,stepc2,stepn1,dt,params)
    ex30,dex30 = s3rads3triex30(stepn,stepc1,stepc2,stepn1,dt,params)
    ex31,dex31 = s3rads3triex31(stepn,stepc1,stepc2,stepn1,dt,params)
    ex32,dex32 = s3rads3triex32(stepn,stepc1,stepc2,stepn1,dt,params)
    ex33,dex33 = s3rads3triex33(stepn,stepc1,stepc2,stepn1,dt,params)
    ex34,dex34 = s3rads3triex34(stepn,stepc1,stepc2,stepn1,dt,params)
    ex35,dex35 = s3rads3triex35(stepn,stepc1,stepc2,stepn1,dt,params)
    ex36,dex36 = s3rads3triex36(stepn,stepc1,stepc2,stepn1,dt,params)

    # v2
    ex37,dex37 = s3rads3triex37(stepn,stepc1,stepc2,stepn1,dt,params)
    ex38,dex38 = s3rads3triex38(stepn,stepc1,stepc2,stepn1,dt,params)
    ex39,dex39 = s3rads3triex39(stepn,stepc1,stepc2,stepn1,dt,params)
    ex40,dex40 = s3rads3triex40(stepn,stepc1,stepc2,stepn1,dt,params)
    ex41,dex41 = s3rads3triex41(stepn,stepc1,stepc2,stepn1,dt,params)
    ex42,dex42 = s3rads3triex42(stepn,stepc1,stepc2,stepn1,dt,params)
    ex43,dex43 = s3rads3triex43(stepn,stepc1,stepc2,stepn1,dt,params)
    ex44,dex44 = s3rads3triex44(stepn,stepc1,stepc2,stepn1,dt,params)
    ex45,dex45 = s3rads3triex45(stepn,stepc1,stepc2,stepn1,dt,params)

    # v3
    ex46,dex46 = s3rads3triex46(stepn,stepc1,stepc2,stepn1,dt,params)
    ex47,dex47 = s3rads3triex47(stepn,stepc1,stepc2,stepn1,dt,params)
    ex48,dex48 = s3rads3triex48(stepn,stepc1,stepc2,stepn1,dt,params)
    ex49,dex49 = s3rads3triex49(stepn,stepc1,stepc2,stepn1,dt,params)
    ex50,dex50 = s3rads3triex50(stepn,stepc1,stepc2,stepn1,dt,params)
    ex51,dex51 = s3rads3triex51(stepn,stepc1,stepc2,stepn1,dt,params)
    ex52,dex52 = s3rads3triex52(stepn,stepc1,stepc2,stepn1,dt,params)
    ex53,dex53 = s3rads3triex53(stepn,stepc1,stepc2,stepn1,dt,params)
    ex54,dex54 = s3rads3triex54(stepn,stepc1,stepc2,stepn1,dt,params)

    # lam
    ex55,dex55 = s3rads3triex55(stepn,stepc1,stepc2,stepn1,dt,params)
    ex56,dex56 = s3rads3triex56(stepn,stepc1,stepc2,stepn1,dt,params)
    ex57,dex57 = s3rads3triex57(stepn,stepc1,stepc2,stepn1,dt,params)
    ex58,dex58 = s3rads3triex58(stepn,stepc1,stepc2,stepn1,dt,params)
    ex59,dex59 = s3rads3triex59(stepn,stepc1,stepc2,stepn1,dt,params)
    ex60,dex60 = s3rads3triex60(stepn,stepc1,stepc2,stepn1,dt,params)
    ex61,dex61 = s3rads3triex61(stepn,stepc1,stepc2,stepn1,dt,params)
    ex62,dex62 = s3rads3triex62(stepn,stepc1,stepc2,stepn1,dt,params)
    ex63,dex63 = s3rads3triex63(stepn,stepc1,stepc2,stepn1,dt,params)

    jacobian = np.array([
        dex1,dex2,dex3,
        dex4,dex5,dex6,
        dex7,dex8,dex9,

        dex10,dex11,dex12,
        dex13,dex14,dex15,
        dex16,dex17,dex18,

        dex19,dex20,dex21,
        dex22,dex23,dex24,
        dex25,dex26,dex27,

        dex28,dex29,dex30,
        dex31,dex32,dex33,
        dex34,dex35,dex36,

        dex37,dex38,dex39,
        dex40,dex41,dex42,
        dex43,dex44,dex45,

        dex46,dex47,dex48,
        dex49,dex50,dex51,
        dex52,dex53,dex54,

        dex55,dex56,dex57,
        dex58,dex59,dex60,
        dex61,dex62,dex63

    ])

    conlist = np.array([
        ex1,ex2,ex3,
        ex4,ex5,ex6,
        ex7,ex8,ex9,

        ex10,ex11,ex12,
        ex13,ex14,ex15,
        ex16,ex17,ex18,

        ex19,ex20,ex21,
        ex22,ex23,ex24,
        ex25,ex26,ex27,

        ex28,ex29,ex30,
        ex31,ex32,ex33,
        ex34,ex35,ex36,

        ex37,ex38,ex39,
        ex40,ex41,ex42,
        ex43,ex44,ex45,

        ex46,ex47,ex48,
        ex49,ex50,ex51,
        ex52,ex53,ex54,

        ex55,ex56,ex57,
        ex58,ex59,ex60,
        ex61,ex62,ex63

    ])

    diff1 = np.linalg.solve(jacobian,-conlist)

    val1 = diff1 + initvec

    # Begin Iterations
    counter = 0
    while (np.linalg.norm(diff1) >= tol and counter <= imax):
        stepc1 = val1[0::3].copy()
        stepc2 = val1[1::3].copy()
        stepn1 = val1[2::3].copy()

        # p1
        ex1,dex1 = s3rads3triex1(stepn,stepc1,stepc2,stepn1,dt,params)
        ex2,dex2 = s3rads3triex2(stepn,stepc1,stepc2,stepn1,dt,params)
        ex3,dex3 = s3rads3triex3(stepn,stepc1,stepc2,stepn1,dt,params)
        ex4,dex4 = s3rads3triex4(stepn,stepc1,stepc2,stepn1,dt,params)
        ex5,dex5 = s3rads3triex5(stepn,stepc1,stepc2,stepn1,dt,params)
        ex6,dex6 = s3rads3triex6(stepn,stepc1,stepc2,stepn1,dt,params)
        ex7,dex7 = s3rads3triex7(stepn,stepc1,stepc2,stepn1,dt,params)
        ex8,dex8 = s3rads3triex8(stepn,stepc1,stepc2,stepn1,dt,params)
        ex9,dex9 = s3rads3triex9(stepn,stepc1,stepc2,stepn1,dt,params)

        # p2
        ex10,dex10 = s3rads3triex10(stepn,stepc1,stepc2,stepn1,dt,params)
        ex11,dex11 = s3rads3triex11(stepn,stepc1,stepc2,stepn1,dt,params)
        ex12,dex12 = s3rads3triex12(stepn,stepc1,stepc2,stepn1,dt,params)
        ex13,dex13 = s3rads3triex13(stepn,stepc1,stepc2,stepn1,dt,params)
        ex14,dex14 = s3rads3triex14(stepn,stepc1,stepc2,stepn1,dt,params)
        ex15,dex15 = s3rads3triex15(stepn,stepc1,stepc2,stepn1,dt,params)
        ex16,dex16 = s3rads3triex16(stepn,stepc1,stepc2,stepn1,dt,params)
        ex17,dex17 = s3rads3triex17(stepn,stepc1,stepc2,stepn1,dt,params)
        ex18,dex18 = s3rads3triex18(stepn,stepc1,stepc2,stepn1,dt,params)

        #p3
        ex19,dex19 = s3rads3triex19(stepn,stepc1,stepc2,stepn1,dt,params)
        ex20,dex20 = s3rads3triex20(stepn,stepc1,stepc2,stepn1,dt,params)
        ex21,dex21 = s3rads3triex21(stepn,stepc1,stepc2,stepn1,dt,params)
        ex22,dex22 = s3rads3triex22(stepn,stepc1,stepc2,stepn1,dt,params)
        ex23,dex23 = s3rads3triex23(stepn,stepc1,stepc2,stepn1,dt,params)
        ex24,dex24 = s3rads3triex24(stepn,stepc1,stepc2,stepn1,dt,params)
        ex25,dex25 = s3rads3triex25(stepn,stepc1,stepc2,stepn1,dt,params)
        ex26,dex26 = s3rads3triex26(stepn,stepc1,stepc2,stepn1,dt,params)
        ex27,dex27 = s3rads3triex27(stepn,stepc1,stepc2,stepn1,dt,params)

        # v1
        ex28,dex28 = s3rads3triex28(stepn,stepc1,stepc2,stepn1,dt,params)
        ex29,dex29 = s3rads3triex29(stepn,stepc1,stepc2,stepn1,dt,params)
        ex30,dex30 = s3rads3triex30(stepn,stepc1,stepc2,stepn1,dt,params)
        ex31,dex31 = s3rads3triex31(stepn,stepc1,stepc2,stepn1,dt,params)
        ex32,dex32 = s3rads3triex32(stepn,stepc1,stepc2,stepn1,dt,params)
        ex33,dex33 = s3rads3triex33(stepn,stepc1,stepc2,stepn1,dt,params)
        ex34,dex34 = s3rads3triex34(stepn,stepc1,stepc2,stepn1,dt,params)
        ex35,dex35 = s3rads3triex35(stepn,stepc1,stepc2,stepn1,dt,params)
        ex36,dex36 = s3rads3triex36(stepn,stepc1,stepc2,stepn1,dt,params)

        # v2
        ex37,dex37 = s3rads3triex37(stepn,stepc1,stepc2,stepn1,dt,params)
        ex38,dex38 = s3rads3triex38(stepn,stepc1,stepc2,stepn1,dt,params)
        ex39,dex39 = s3rads3triex39(stepn,stepc1,stepc2,stepn1,dt,params)
        ex40,dex40 = s3rads3triex40(stepn,stepc1,stepc2,stepn1,dt,params)
        ex41,dex41 = s3rads3triex41(stepn,stepc1,stepc2,stepn1,dt,params)
        ex42,dex42 = s3rads3triex42(stepn,stepc1,stepc2,stepn1,dt,params)
        ex43,dex43 = s3rads3triex43(stepn,stepc1,stepc2,stepn1,dt,params)
        ex44,dex44 = s3rads3triex44(stepn,stepc1,stepc2,stepn1,dt,params)
        ex45,dex45 = s3rads3triex45(stepn,stepc1,stepc2,stepn1,dt,params)

        # v3
        ex46,dex46 = s3rads3triex46(stepn,stepc1,stepc2,stepn1,dt,params)
        ex47,dex47 = s3rads3triex47(stepn,stepc1,stepc2,stepn1,dt,params)
        ex48,dex48 = s3rads3triex48(stepn,stepc1,stepc2,stepn1,dt,params)
        ex49,dex49 = s3rads3triex49(stepn,stepc1,stepc2,stepn1,dt,params)
        ex50,dex50 = s3rads3triex50(stepn,stepc1,stepc2,stepn1,dt,params)
        ex51,dex51 = s3rads3triex51(stepn,stepc1,stepc2,stepn1,dt,params)
        ex52,dex52 = s3rads3triex52(stepn,stepc1,stepc2,stepn1,dt,params)
        ex53,dex53 = s3rads3triex53(stepn,stepc1,stepc2,stepn1,dt,params)
        ex54,dex54 = s3rads3triex54(stepn,stepc1,stepc2,stepn1,dt,params)

        # lam
        ex55,dex55 = s3rads3triex55(stepn,stepc1,stepc2,stepn1,dt,params)
        ex56,dex56 = s3rads3triex56(stepn,stepc1,stepc2,stepn1,dt,params)
        ex57,dex57 = s3rads3triex57(stepn,stepc1,stepc2,stepn1,dt,params)
        ex58,dex58 = s3rads3triex58(stepn,stepc1,stepc2,stepn1,dt,params)
        ex59,dex59 = s3rads3triex59(stepn,stepc1,stepc2,stepn1,dt,params)
        ex60,dex60 = s3rads3triex60(stepn,stepc1,stepc2,stepn1,dt,params)
        ex61,dex61 = s3rads3triex61(stepn,stepc1,stepc2,stepn1,dt,params)
        ex62,dex62 = s3rads3triex62(stepn,stepc1,stepc2,stepn1,dt,params)
        ex63,dex63 = s3rads3triex63(stepn,stepc1,stepc2,stepn1,dt,params)

        jacobian = np.array([
            dex1,dex2,dex3,
            dex4,dex5,dex6,
            dex7,dex8,dex9,

            dex10,dex11,dex12,
            dex13,dex14,dex15,
            dex16,dex17,dex18,

            dex19,dex20,dex21,
            dex22,dex23,dex24,
            dex25,dex26,dex27,

            dex28,dex29,dex30,
            dex31,dex32,dex33,
            dex34,dex35,dex36,

            dex37,dex38,dex39,
            dex40,dex41,dex42,
            dex43,dex44,dex45,

            dex46,dex47,dex48,
            dex49,dex50,dex51,
            dex52,dex53,dex54,

            dex55,dex56,dex57,
            dex58,dex59,dex60,
            dex61,dex62,dex63

        ])

        conlist = np.array([
            ex1,ex2,ex3,
            ex4,ex5,ex6,
            ex7,ex8,ex9,

            ex10,ex11,ex12,
            ex13,ex14,ex15,
            ex16,ex17,ex18,

            ex19,ex20,ex21,
            ex22,ex23,ex24,
            ex25,ex26,ex27,

            ex28,ex29,ex30,
            ex31,ex32,ex33,
            ex34,ex35,ex36,

            ex37,ex38,ex39,
            ex40,ex41,ex42,
            ex43,ex44,ex45,

            ex46,ex47,ex48,
            ex49,ex50,ex51,
            ex52,ex53,ex54,

            ex55,ex56,ex57,
            ex58,ex59,ex60,
            ex61,ex62,ex63

        ])

        diff2 = np.linalg.solve(jacobian,-conlist)

        val2 = diff2 + val1

        val1 = val2
        diff1 = diff2
        counter += 1

    return val1[2::3]




