import core
import numpy as np
import matplotlib.pyplot as plt

### Ilic Replication
Nx = 181 # don't keep this too low: the boundary epsilons are not very representative of a real cell. Nx=181 is good
Ny = 1
nG = 30
beta = 0.02

# Defining Ilic dimensions
d = 1.8  # unit cell width
x1 = 0.25*d # positions of blocks in unit cell
x2 = 0.85*d 
w1 = 0.35*d # width of blocks in unit cell, 0.45d works for good optimum
w2 = 0.15*d

h = 0.5  # thickness of resonator layer
t = 0.5  # also equal to thickness of substrate layer
v_width = 2

E_vacuum = 1.
E_Si = 3.5**2    # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)
E_SiO2 = 1.45**2  # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)


# Lattice constants. I will consider a lattice with square-shaped unit cells of size d x d (in natural units specified above)
L1 = [d, 0]
wavelength = 1.5
c=1.
freq = c/wavelength

'''
Ilic replication
'''

orders = [-1,0,1]
thetas = np.linspace(-20*np.pi/180, 20*np.pi/180, 81)
Ris = np.empty((len(thetas), len(orders)))
Tis = np.empty((len(thetas), len(orders)))


cell_geometry, eps1, eps2 = core.linear_boundary_geometry(Nx,d,x1,x2,w1,w2)
print(eps1,eps2)
for j in range(0,len(thetas)):
    theta = thetas[j]
    Rs,Ts = core.grcwa_reflectance_transmittance_orders(nG,orders,Nx,d,theta*180/np.pi,freq,x1,x2,w1,w2)
    # Construct matrix with jth row being the
    Ris[j,:] = Rs
    Tis[j,:] = Ts
    
    print("Incidence: {:g}".format(theta*180/np.pi))







legend = []
colors = ['red', 'grey', 'blue']
ax = plt.gca()
for i in range(0,len(orders)):
    order = orders[i]
    # color=next(ax._get_lines.prop_cycler)['color']
    color = colors[i]
    plt.plot(thetas*180/np.pi, Ris[:,i], color=color)
    legend.append(r"$r_{}$".format("{" + str(orders[i]) + "}"))
    plt.plot(thetas*180/np.pi, Tis[:,i], '--', color=color)
    legend.append(r"$t_{}$".format("{" + str(orders[i]) + "}"))

plt.legend(legend)
plt.xlabel(r"Incident angle ($^\circ$)")
plt.ylabel("Reflection/transmission")
plt.title("Reflection and transmission coefficients for asymmetric Ilic-style grating")
plt.ylim([0,1.])

plt.savefig('grcwa/ilic_replication/figs/ilic_GRCWA_rtmodes_nG{}'.format(nG))
plt.show()
