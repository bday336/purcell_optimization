from numpy import cos,sin,array

def purcell2dfunc(statevec,params):
	a1,a2,ad1,ad2,lamx,lamy,lamz = statevec
	l,nu,sr,kt = params