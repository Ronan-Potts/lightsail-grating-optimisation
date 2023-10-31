import core
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('D:\GitHub Repositories\PHYS3888\lightsail-grating-optimisation\grcwa\plots_A4_scale.mplstyle')

## Discretisation values
nG = 30
Nx = 1801

## Ilic cell parameters
d = 1.8
x1 = 0.25*d
x2 = 0.85*d
w1 = 0.35*d
w2 = 0.15*d

# Permittivities
E_Si = 3.5**2
E_SiO2 = 1.45**2
E_vacuum = 1.

## Light incidence
c = 1.
wavelength = 1.5
freq = c/wavelength
thetas = np.linspace(-20,20,100)

R_plus_T = []
for theta in thetas:
    print("Angle of incidence: {} deg".format(round(theta,3)))
    Rs, Ts = core.grcwa_reflectance_transmittance_orders(nG,[0,1,-1],Nx,d,theta,freq,x1,x2,w1,w2)
    R_plus_T.append(np.sum(Rs)+np.sum(Ts))

plt.plot(thetas,R_plus_T)
plt.xlabel("Incidence angle [deg]")
plt.ylabel(r"$R+T=1$")
plt.show()