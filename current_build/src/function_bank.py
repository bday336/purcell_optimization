# Function Bank

import numpy as np

def se2cay(vx,vy,wth,dt):
    v1,v2,v3 = wth*dt, vx*dt, vy*dt
    return np.array([
        [(4. - v1**2.)/(4.+v1**2.) , (-4.*v1)/(4.+(dt*wth)**2.) , (4.*v2 - 2.*v3*v1)/(4.+v1**2.)],
        [(4.*v1)/(4.+v1**2.) , (4. - v1**2.)/(4.+(dt*wth)**2.)  , (4.*v3 + 2.*v2*v1)/(4.+v1**2.)],
        [0. , 0. , 1.]
    ])

def se2group(x,y,th):
    return np.array([
        [np.cos(th),-np.sin(th),x],
        [np.sin(th),np.cos(th),y],
        [0. , 0. , 1.]
    ])
