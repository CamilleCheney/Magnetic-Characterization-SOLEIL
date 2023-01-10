import matplotlib.pyplot as plt
import numpy as np
import radia as rad
from math import pi, sin, cos

#########################    Dimensions   #####################################
# Quadrupole
r1 = 6 #inner radius
r2 = 17 #outer radius
a = pi/8 #angle
L = 81.1 #length

# Magnetic rod
R = 26.5 #mm
O = pi/4

#Colors
red = [1,0,0]
blue = [0,0.5,1]
green = [0,1,0.4]



#######################   Materials   #########################################

#(Permanent) Magnet material: NdFeB with 1.26 Tesla Remanent Magnetization
magnet_mat = rad.MatStd('NdFeB', 1.26)
iron_mat = rad.MatStd('AFK1', 2.35)




##########################   Build the geometry   #############################

#Function that build 1/4 of the quadrupole
def Geom2(N):
    #1st part = 1/16 quadrupole
    a = 0
    XY_1 = []
    XY_2 = []
    Mesh = []
    for i in range(N+1):
        x1, y1 = r1*np.cos(a), r1*np.sin(a)
        x2, y2 = r2*np.cos(a), r2*np.sin(a)
        XY_1.append([x1, y1])
        XY_2.append([x2, y2])
        Mesh.append([[1,1], [1,1]])
        da = np.pi/8/N 
        a += da        
    XY = XY_1 + XY_2[::-1]
    Magnetization = [0,0,1]    
    quad_1 = rad.ObjMltExtTri(0, L, XY, Mesh, Magnetization)
    #Apply Color to it
    rad.ObjDrwAtr(quad_1, red)
    #Apply material to object
    rad.MatApl(quad_1, magnet_mat)        
    
    
    #2nd part = 1/8 quadrupole
    a = np.pi/8
    XY_1 = []
    XY_2 = []
    Mesh = []
    for i in range(2*N+1):
        x1, y1 = r1*np.cos(a), r1*np.sin(a)
        x2, y2 = r2*np.cos(a), r2*np.sin(a)
        XY_1.append([x1, y1])
        XY_2.append([x2, y2])
        Mesh.append([[1,1], [1,1]])
        da = np.pi/8/N 
        a += da        
    XY = XY_1 + XY_2[::-1]
    quad_2 = rad.ObjMltExtTri(0, L, XY, Mesh)
    #Apply Color to it
    rad.ObjDrwAtr(quad_2, blue)
    #Apply material to object
    rad.MatApl(quad_2, iron_mat) 
    
    
    #3rd part = 1/16 quadrupole
    a = 3*np.pi/8
    XY_1 = []
    XY_2 = []
    Mesh = []
    for i in range(N+1):
        x1, y1 = r1*np.cos(a), r1*np.sin(a)
        x2, y2 = r2*np.cos(a), r2*np.sin(a)
        XY_1.append([x1, y1])
        XY_2.append([x2, y2])
        Mesh.append([[1,1], [1,1]])
        da = np.pi/8/N 
        a += da        
    XY = XY_1 + XY_2[::-1]
    Magnetization = [0,1,0]    
    quad_3 = rad.ObjMltExtTri(0, L, XY, Mesh, Magnetization)
    #Apply Color to it
    rad.ObjDrwAtr(quad_3, red)
    #Apply material to object
    rad.MatApl(quad_3, magnet_mat) 

    #Gather the 3 elements of 1/
    quad = rad.ObjCnt([quad_1, quad_2, quad_3])
    return quad
###############################################################################

#Another function that build the same geometry but in another way
#Function that build 1/4 of the quadrupole
def Geom1(N):
    
    part1 = []
    part2 = []
    part3 = []
    #1st part = 1/16 quadrupole
    a = 0
    for i in range(N):
        
        da = np.pi/8/N 
        x1, y1 = r1*np.cos(a), r1*np.sin(a)
        x2, y2 = r2*np.cos(a), r2*np.sin(a)
        x3, y3 = r1*np.cos(a+da), r1*np.sin(a+da)
        x4, y4 = r2*np.cos(a+da), r2*np.sin(a+da)
        Nodes = [[x1,y1],[x2,y2],[x4,y4],[x3,y3]]
        Mesh = [[1,1],[1,1],[1,1],[1,1]]
        Magnetization = [0,0,-1]
        g1 = rad.ObjMltExtTri(0, L, Nodes, Mesh, Magnetization)
        part1.append(g1)
        a += da
    pole1 = rad.ObjCnt(part1)
    #Apply Color to it
    rad.ObjDrwAtr(pole1, red)
    #Apply material to object
    rad.MatApl(pole1, magnet_mat)

    
    #2nd part = 1/8 quadrupole
    a = np.pi/8
    for i in range(2*N):
        da = np.pi/8/N 
        x1, y1 = r1*np.cos(a), r1*np.sin(a)
        x2, y2 = r2*np.cos(a), r2*np.sin(a)
        x3, y3 = r1*np.cos(a+da), r1*np.sin(a+da)
        x4, y4 = r2*np.cos(a+da), r2*np.sin(a+da)
        Nodes = [[x1,y1],[x2,y2],[x4,y4],[x3,y3]]
        Mesh = [[1,1],[1,1],[1,1],[1,1]]
        g2 = rad.ObjMltExtTri(0, L, Nodes, Mesh)
        part2.append(g2)
        a += da
    pole2 = rad.ObjCnt(part2)
    #Apply Color to it
    rad.ObjDrwAtr(pole2, blue)
    #Apply material to object
    rad.MatApl(pole2, iron_mat)

    
    #3rd part = 1/16 quadrupole
    a = 3*np.pi/8
    for i in range(N):
        da = np.pi/8/N 
        x1, y1 = r1*np.cos(a), r1*np.sin(a)
        x2, y2 = r2*np.cos(a), r2*np.sin(a)
        x3, y3 = r1*np.cos(a+da), r1*np.sin(a+da)
        x4, y4 = r2*np.cos(a+da), r2*np.sin(a+da)
        Nodes = [[x1,y1],[x2,y2],[x4,y4],[x3,y3]]
        Mesh = [[1,1],[1,1],[1,1],[1,1]]
        Magnetization = [0,0,-1]
        g3 = rad.ObjMltExtTri(0, L, Nodes, Mesh, Magnetization)
        part3.append(g3)
        a += da
    pole3 = rad.ObjCnt(part3)
    #Apply Color to it
    rad.ObjDrwAtr(pole3, red)
    #Apply material to object
    rad.MatApl(pole3, magnet_mat)

    #Gather the 3 elements of 1/
    quad = rad.ObjCnt([pole1, pole2, pole3])
    return quad
###############################################################################


#Function that build one permanent magnet with a cylinder shape
def Mag_rod(y, z, mx, my, mz, color):
    r = 7.5
    Cyl = rad.ObjCylMag([0, y, z], r, L, 20, 'x', [mx, my, mz])
    rad.ObjDrwAtr(Cyl, color)
    rad.MatApl(Cyl, magnet_mat)

    return Cyl
###############################################################################




#######################   Build the entire magnet   ###########################

# Build 1/4 of the quadrupole with N subdivison
N_subdivision = 5
Quadrupole = Geom2(N_subdivision)


#B uild one rod parmanent magnet
M = pi/4 #Modulation angle
Rod = Mag_rod(R*cos(O), R*sin(O), 0, cos(M), sin(M), green) 


#Segmentation in the [x,y,z] direction of the 1/4 quadrupole
rad.ObjDivMag(Quadrupole, [5,1,1])


#Segmentation in the [x,y,z] direction of the rod
rad.ObjDivMag(Rod, [5,4,4])


#Gather the rod and the 1/4 quadrupole
QUAPEVA =rad.ObjCnt([Quadrupole, Rod]) 


#Define the symmetries in order to build the whole magnet
rad.TrfZerPara(QUAPEVA,[0,0,0],[0,0,1])
rad.TrfZerPara(QUAPEVA,[0,0,0],[0,1,0])


#Display the Geometry in 3D Viewer
rad.ObjDrwOpenGL(QUAPEVA)





#######################   Compute/solve B field   #############################
rad.Solve(QUAPEVA,0.001,1000) 



#######################   Radial field B_z(y)   ###############################
#Calculate Magnetic Field
Bz_y = np.array(rad.FldLst(QUAPEVA, 'Bz', [0,-r1,0], [0,r1,0], 100,'arg', -r1))
#Plot B_z field along the y axis
plt.plot(Bz_y[:,0][1:-1], Bz_y[:,1][1:-1], "b")
plt.xlabel("y position [mm]")
plt.ylabel("$B_z$ [T]")
plt.title("$B_z(y)$")
plt.grid()
plt.savefig('Bz(y).pdf', dpi=1080)
plt.show()

#######################   Radial field integral  B_z(y)   #####################
#Load the data we measured
Bz_y_int_exp, y, z = np.loadtxt('Field_Integral_Bz(y).txt', unpack=True, skiprows=1)

Y = np.linspace(-r1+1, r1-1, 100)
Bz_y_int = []
for i in range(len(Y)):
    Bz_y_int.append(rad.FldInt(QUAPEVA, 'inf', 'Bz', [-1,Y[i],0], [1,Y[i],0] ))
plt.plot(Y, Bz_y_int, 'b', label='Simulation')
plt.plot(y, Bz_y_int_exp, 'rx', label='Measurements')
plt.xlabel("y position [mm]")
plt.ylabel("$B$ integral [T.mm]")
plt.title("Field Integral $B_z(y)$")
plt.grid()
plt.legend()
plt.savefig('Field_Int_Bz(y).pdf', dpi=1080)
plt.show()




#######################   Radial field B_y(z)   ###############################
#Calculate Magnetic Field
By_z = np.array(rad.FldLst(QUAPEVA, 'By', [0,0,-r1], [0,0,r1], 100,'arg', -r1))
#Plot B_y field along the z axis
plt.plot(-By_z[:,0][1:-1], By_z[:,1][1:-1], "b")
plt.xlabel("z position [mm]")
plt.ylabel("$B_y$ [T]")
plt.title("$B_y(z)$")
plt.grid()
plt.savefig('By(z).pdf', dpi=1080)
plt.show()

#######################   Radial field integral  B_y(z)   #####################
By_z_int_exp, z, y = np.loadtxt('Field_Integral_By(z).txt', unpack=True, skiprows=1)

Z = np.linspace(-r1+1, r1-1, 100)
By_z_int = []
for i in range(len(Z)):
    By_z_int.append(rad.FldInt(QUAPEVA, 'inf', 'By', [-1,0,Z[i]], [1,0,Z[i]] ))
plt.plot(-Z, By_z_int, 'b', label='Simulation')
plt.plot(z, By_z_int_exp, 'rx', label='Measurements')
plt.xlabel("z position [mm]")
plt.ylabel("$B$ integral [T.mm]")
plt.title("Field Integral $B_y(z)$")
plt.grid()
plt.legend()
plt.savefig('Field_Int_By(z).pdf', dpi=1080)
plt.show()




#######################   Circular field integral  B_theta   ##################
#Import data
Y, Z, B_circ_int_exp, TrajAngle = np.loadtxt('Field_Integral_circle.txt', unpack=True, skiprows=1)

Y = Y - (max(Y)+min(Y))/2 #To center Y in 0 in the simulation
a = np.linspace(0, 2*pi, len(Y)) #Angles from 0 to 2*pi

B_circ_int = []
# Loop which calculate B field along a circle of radius R
for i in  range(len(Y)):
    By_int = rad.FldInt(QUAPEVA, 'inf', 'iby', [-1,Y[i],Z[i]], [1,Y[i],Z[i]] )
    Bz_int = rad.FldInt(QUAPEVA, 'inf', 'ibz', [-1,Y[i],Z[i]], [1,Y[i],Z[i]] )

    #Here is the projection of By_int and Bz_int on B_circ_int
    B_circ_int.append( Bz_int*np.sin(a[i]) + By_int*np.cos(a[i]) )
    
    
plt.plot(a, B_circ_int_exp, 'r:.', label='Measurements')
plt.plot(a, B_circ_int, 'b', label='Simulation')
plt.title('Field integral along a circle of radius r=%.2f [mm]' %Y[0])
plt.ylabel('B integral [T.mm]')
plt.xlabel('$\\theta$ angle [rad]')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Field_Int_B_circle.pdf', dpi=1080)
plt.show()



#######################   Plot field integral in polar coordinates   ##########
fig = plt.figure(figsize=[7,4])
ax1 = plt.subplot(121, projection='polar')
ax2 = plt.subplot(122, projection='polar')
ax1.plot(a, B_circ_int_exp, 'r:.', label='Measurements')
ax1.plot(a, B_circ_int, 'b', label='Simulation')
ax1.set_title('B integral')
ax1.legend()
ax2.plot(a, np.abs(B_circ_int_exp), 'r:.', label='Measurements')
ax2.plot(a, np.abs(B_circ_int), 'b', label='Simulation')
ax2.set_title('|B| integral')
ax2.legend()
fig.suptitle('Field integral along a circle of radius r=%.2f [mm]' %Y[0])
plt.tight_layout()
plt.savefig('Field_Int_B_circle_polar.pdf', dpi=1080)
plt.show()    