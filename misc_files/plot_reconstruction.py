# Generic Simulation Script

import sys
import numpy as np
import matplotlib.pyplot as plt
from current_build.src.test_system_simulations.purcell2dj2a import purcell2dj2a

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

# Load data
data0 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang{}_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]))
# data1 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang1_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data2 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang2_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data3 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang3_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data4 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang4_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data5 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang5_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data6 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang6_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data7 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang7_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data8 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang8_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data9 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang9_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data10 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang10_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data11 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang11_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data12 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang12_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data13 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang13_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data14 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang14_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# data15 = np.load("l{}{}{}/purcell2d_lx{}_ly{}_lz{}_ang15_tmax{}_dt001.npy".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))

# Calculate the motion along fiber

# Data containers
link2veldat0 = []

caydat0 = []

fulldat0 = []

# Populate data containers
for a in range(data0.shape[0]):
    link2veldat0.append(np.asarray(purcell2dj2a(data0[a,0],data0[a,1],.01)) @ np.array([data0[a,2],data0[a,3]]))

for b in range(data0.shape[0]):
    caydat0.append(se2cay(link2veldat0[b][0],link2veldat0[b][1],link2veldat0[b][2],.001))

fulldat0.append(np.identity(3))

for c in range(data0.shape[0]):
    fulldat0.append(fulldat0[-1] @ caydat0[c])

# phase plot of angles
fig,((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9))=plt.subplots(3,3,figsize=(16,8))
fig.tight_layout()

ax1.set_title("Angles vs. Time")
ax1.plot(np.arange(0,float(sys.argv[5])+.001,.001),data0[:,0],label="al1")
ax1.plot(np.arange(0,float(sys.argv[5])+.001,.001),data0[:,1],label="al2")
ax2.set_title("Angle Velocities vs. Time")
ax2.plot(np.arange(0,float(sys.argv[5])+.001,.001),data0[:,2],label="ald1")
ax2.plot(np.arange(0,float(sys.argv[5])+.001,.001),data0[:,3],label="ald2")
ax3.text(0.5, 0.5, 'l{}{}{} Ang {}'.format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]), horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize='x-large')

ax4.set_title("Angle 1 vs. Angle 2")
ax4.plot(data0[:,0],data0[:,1],label="al1")
ax5.set_title("dAngle 1 vs. dAngle 2")
ax5.plot(data0[:,2],data0[:,3],label="ald1")

# ax7.set_title("xi_x")
# ax7.plot(np.arange(0,float(sys.argv[5])+.001,.001),np.asarray(link2veldat0)[:,0],label="xi_x")
# ax8.set_title("xi_y")
# ax8.plot(np.arange(0,float(sys.argv[5])+.001,.001),np.asarray(link2veldat0)[:,1],label="xi_y")
# ax9.set_title("xi_th")
# ax9.plot(np.arange(0,float(sys.argv[5])+.001,.001),np.asarray(link2veldat0)[:,2],label="xi_th")

ax7.set_title("x")
ax7.plot(np.arange(0,float(sys.argv[5])+.001,.001),np.asarray(caydat0)[:,0,2],label="x")
ax8.set_title("y")
ax8.plot(np.arange(0,float(sys.argv[5])+.001,.001),np.asarray(caydat0)[:,1,2],label="y")
ax9.set_title("th")
ax9.plot(np.arange(0,float(sys.argv[5])+.001,.001),np.arcsin(np.asarray(caydat0)[:,1,0]),label="th")


# ax1.plot(data0[:,0],data0[:,1],label="ang0")
# ax1.plot(data1[:,0],data1[:,1],label="ang1")
# ax1.plot(data2[:,0],data2[:,1],label="ang2")
# ax1.plot(data3[:,0],data3[:,1],label="ang3")
# ax1.plot(data4[:,0],data4[:,1],label="ang4")
# ax1.plot(data5[:,0],data5[:,1],label="ang5")
# ax1.plot(data6[:,0],data6[:,1],label="ang6")
# ax1.plot(data7[:,0],data7[:,1],label="ang7")
# ax1.plot(data8[:,0],data8[:,1],label="ang8")
# ax1.plot(data9[:,0],data9[:,1],label="ang9")
# ax1.plot(data10[:,0],data10[:,1],label="ang10")
# ax1.plot(data11[:,0],data11[:,1],label="ang11")
# ax1.plot(data12[:,0],data12[:,1],label="ang12")
# ax1.plot(data13[:,0],data13[:,1],label="ang13")
# ax1.plot(data14[:,0],data14[:,1],label="ang14")
# ax1.plot(data15[:,0],data15[:,1],label="ang15")

# ax2.plot(data0[:,2],data0[:,3],label="ang0")
# ax2.plot(data1[:,2],data1[:,3],label="ang1")
# ax2.plot(data2[:,2],data2[:,3],label="ang2")
# ax2.plot(data3[:,2],data3[:,3],label="ang3")
# ax2.plot(data4[:,2],data4[:,3],label="ang4")
# ax2.plot(data5[:,2],data5[:,3],label="ang5")
# ax2.plot(data6[:,2],data6[:,3],label="ang6")
# ax2.plot(data7[:,2],data7[:,3],label="ang7")
# ax2.plot(data8[:,2],data8[:,3],label="ang8")
# ax2.plot(data9[:,2],data9[:,3],label="ang9")
# ax2.plot(data10[:,2],data10[:,3],label="ang10")
# ax2.plot(data11[:,2],data11[:,3],label="ang11")
# ax2.plot(data12[:,2],data12[:,3],label="ang12")
# ax2.plot(data13[:,2],data13[:,3],label="ang13")
# ax2.plot(data14[:,2],data14[:,3],label="ang14")
# ax2.plot(data15[:,2],data15[:,3],label="ang15")
# ax.set_xlabel('t')
# ax.set_ylabel('l')
# ax1.legend()
# ax2.legend()

# # ax3.legend()
# ax4.legend()
# ax5.legend()
# ax6.legend()
# ax7.legend()
# ax8.legend()
# ax9.legend()

# plt.savefig('recon_data_l{}{}{}_tmax{}.png'.format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
plt.show()