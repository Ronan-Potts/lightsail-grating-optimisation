import core
import numpy as np
# Truncation order
nG = 30
# Resolution 
Nx = 180001

# Permittivities
E_vacuum = 1.
E_Si = 3.5**2    # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)
E_SiO2 = 1.45**2  # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)


## Incident light
c = 1.
theta = 0.
wavelength = 1.5
freq = c/wavelength
'''
Lattice cell parameters
'''
# Defining Ilic dimensions
d = 1.8  # unit cell width
x1 = 0.25*d # positions of blocks in unit cell
x2 = 0.85*d 
w1 = 0.35*d # width of blocks in unit cell
w2 = 0.15*d


### Define cell geometry using element position and width only
# cell_geometry, eps1, eps2 = core.linear_boundary_geometry(Nx,d,x1,x2,w1,w2)


### Define cell geometry using permittivity at boundaries
cell_geometry = core.linear_permittivity_geometry(Nx,d,x1,x2,w1,w2,5,5)


### Reflectance and transmittance
R,T = core.grcwa_reflectance_transmittance(nG,cell_geometry,Nx,d,theta,freq)
print("Reflectance: {}, Transmittance: {}".format(R,T))


### Reflectance and transmittance by orders: -1, 0, 1
print("")
orders = [-1,0,1]
Rs,Ts = core.grcwa_reflectance_transmittance_orders(nG,orders,Nx,d,theta,freq,x1,x2,w1,w2)
print("Reflectance at orders [-1,0,1]:", Rs)
print("Transmittance at orders [-1,0,1]:", Ts)


### Efficiencies
print("")
orders = [-1,0,1]
efficiencies = core.grcwa_efficiencies(nG,orders,cell_geometry,Nx,d,theta,freq)
print("Efficiencies:", efficiencies)


### Derivative of efficiencies
print("")
from autograd import grad
En1_fun = lambda theta: core.grcwa_efficiencies(nG,[0,1,-1],cell_geometry,Nx,d,theta,freq)[-1]
E1_fun = lambda theta: core.grcwa_efficiencies(nG,[0,1,-1],cell_geometry ,Nx,d,theta,freq)[1]
pEn1_pTheta = grad(En1_fun)(0.)
pE1_pTheta = grad(E1_fun)(0.)

numerical = (En1_fun(0.005) - En1_fun(-0.005))/ 0.01
print("Partial derivative of e-1:", pEn1_pTheta)
print("Numerical derivative of e-1:", numerical)

