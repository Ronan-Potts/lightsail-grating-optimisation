import grcwa
import numpy as np
from matplotlib import pyplot as plt
'''
Being numerical in nature, I need to define the truncation order of the fourier decomposition as well as the resolution of the unit cell.
'''
# Truncation order
nG = 30
# Resolution 
Nx = 10000
Ny = 1

'''
The goal of this document is to replicate the results in Ilic & Atwater (2019) Fig. 2b.
This paper looks at a metasurface where the unit cell consists of two optical resonators
of subwavelength thickness. The side-view looks like

     w_1      w_2
  <------>   <--->
   ______     ____               ^ z
  |      |   |    |              |
__|      |___|    |___           X-----> x
-----|---------|----->            y
    x_1       x_2

<--------- d -------->

The resonators are made of silicon, while the substrate is SiO2 (silicon dioxide). The
height of both the resonators and the substrate is 0.5 microns. Ilic & Atwater use
d = 1.8 microns; x_1 = 0.25d; x_2 = 0.85d; w_1 = 0.15d; w_2 = 0.35d. 

They also assume an incident beam of lambda = 1.5 microns. Note that natural units
measure distance in units of the Bohr radius. 
'''
# Micron           

# Defining Ilic dimensions
d = 1.8  # unit cell width
dy = 1e-4
x1 = 0.85*d # positions of blocks in unit cell
x2 = 0.25*d 
w1 = 0.15*d # width of blocks in unit cell
w2 = 0.35*d

h = 0.5  # thickness of resonator layer
t = 0.5  # also equal to thickness of substrate layer

wavelength = 1.5
freq = 1./wavelength
thetas = np.linspace(-20*np.pi/180, 20*np.pi/180, 41)
phi = 0.

'''
By convention, GRCWA uses:
        
        vacuum permeability, permitivity and speed of light = 1

Similar to the natural units of particle and atomic physics. I will choose the permitivity of the dielectric material to be 10, and the width of the square unit cell to be 1.5. 
'''
E_vacuum = 1.
E_Si = 3.5**2    # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)
E_SiO2 = 1.45**2  # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)


'''
The lattice is made below.
'''
# Lattice constants. I will consider a lattice with square-shaped unit cells of size d x d (in natural units specified above)
L1 = [d, 0]
L2 = [0, dy]

# Unit cell geometry (rows, cols)
cell_geometry = np.ones((Nx,Ny), dtype=float)*E_vacuum

x0 = np.linspace(0,d,Nx)
y0 = np.linspace(0,dy,Ny)
x, y = np.meshgrid(x0,y0, indexing='ij')
# The design will be completely uniform in the y-direction.
filter = (abs(x-x1) < w1/2) | (abs(x-x2) < w2/2)
cell_geometry[filter] = E_Si

# Visualising the unit cell
# plt.imshow(cell_geometry, interpolation='nearest')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

'''
Setting up the RCWA
                                 _________
                                  Layer 0 (vacuum)
    ___     ______               _________
   |   |   |      |               Layer 1 (pattern)
 __|   |___|      |__            _________
|                    |   ...      Layer 2 (surface)
|____________________|           _________
                                  Layer 3 (vacuum)
                                 _________
'''
Rs = []
Ts = []
RplusTs = []
for theta in thetas:
   print("theta: {:g}".format(theta*180/np.pi))
   obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=0)
   # input layer information
   obj.Add_LayerUniform(1,E_vacuum)     # Layer 0
   obj.Add_LayerGrid(h,Nx,Ny)           # Layer 1
   obj.Add_LayerUniform(h,E_SiO2)   # Layer 2
   obj.Add_LayerUniform(1,E_vacuum)     # Layer 3
   obj.Init_Setup()

   # planewave excitation
   planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
   obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
   # eps in patterned layer
   obj.GridLayer_geteps(cell_geometry.flatten())
   # compute reflection and transmission
   R,T= obj.RT_Solve(normalize=1)
   Rs.append(R)
   Ts.append(T)
   RplusTs.append(R+T)
   # compute reflection and transmission by order
   # Ri(Ti) has length obj.nG, too see which order, check obj.G; too see which kx,ky, check obj.kx obj.ky
   # Ri,Ti= obj.RT_Solve(normalize=1,byorder=1)
   # ords = obj.G # Returns a list of tuples [ord1, ord2] where ord1 is the order in the L1 direction, while ord2 is the order in the L2 direction.


plt.plot(thetas, Rs)
plt.plot(thetas, Ts)
plt.plot(thetas, RplusTs)
plt.legend(['Reflectance', 'Transmittance', 'Sum'])
plt.ylim([0,1.1])

plt.savefig('ISB B/grcwa/ilic_replication/figs/ilic_GRCWA_rt_nG{}'.format(nG))
plt.show()






# Questions:

# 1. What polarisation are we looking at and why? Is it p- or s-polarisation?
#    A. Either works. Laser light is typically polarised. If the sail rotates along the laser beam axis, then the p- and s- polarisations change.

# 2. What is truncation order?

# 3. Is it necessary to sandwich the sail between vacuum layers with this code?

# 4. What is Gmethod?
#    A. See documentation this is for the Fourier space truncation.

# NEXT STEP: DO FOR DIFFERENT ORDERS