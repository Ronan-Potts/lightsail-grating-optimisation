import meep as mp
import math
import numpy as np
import cmath
import matplotlib.pyplot as plt

resolution = 50  # pixels/um

dpml = 0.1  # PML thickness
sz = 1.8 + 2 * dpml
cell_size = mp.Vector3(x = sz, z=sz)
pml_layers = [mp.PML(thickness=dpml,direction=mp.Z,side=mp.High),
              mp.PML(thickness=dpml,direction=mp.Z,side=mp.Low)]

wvl_min = 0.4  # min wavelength
wvl_max = 0.8 # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width
nfreq = 50  # number of frequency bins

# define parameters
d = 1.8
x1 = 0.15 * d
x2 = 0.75 * d
w1 = 0.15 * d
w2 = 0.35 * d
t = 0.5
h = 0.5
epsi_Si = 3.5**2
epsi_SiO2 = 1.45**2
wl = 1.5
silicon = 'silicon'
sio2 = 'SiO2'

#deal with y
custom = 0
offset = 0

def pw_amp(k,x0):
  def _pw_amp(x):
    return cmath.exp(1j*2*math.pi*k.dot(x+x0))
  return _pw_amp

def planar_reflectance(theta):
    # rotation angle (in degrees) of source: CCW around Y axis, 0 degrees along +Z axis
    theta_r = math.radians(theta)

    # plane of incidence is XZ; rotate counter clockwise (CCW) about y-axis
    k = mp.Vector3(z=1/1.5).rotate(mp.Vector3(y=1), theta_r)
    print(k)

    # if normal incidence, force number of dimensions to be 1
    if theta_r == 0:
        dimensions = 1
    else:
        dimensions = 3

    use_cw_solver = False  # CW solver or time stepping?

    src_pt = mp.Vector3(-0.5*sz+dpml+0.3*t,0,-sz/2)
    sources = [mp.Source(mp.ContinuousSource(fcen,fwidth=df) if use_cw_solver else mp.GaussianSource(fcen,fwidth=df),
                        component=mp.Ex,
                        center=src_pt,
                        size=mp.Vector3(0,0,sz),
                        amp_func=pw_amp(k,src_pt))]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        dimensions=dimensions,
        resolution=resolution,
    )

    #where the light firstly touches a surface
    refl_fr = mp.FluxRegion(center=mp.Vector3(z= offset - h/2), size = mp.Vector3(d, 0, 0), direction = mp.Z)
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(z=-0.5 * sz + dpml), 1e-9
        )
    )

    empty_flux = mp.get_fluxes(refl)
    empty_data = sim.get_flux_data(refl)

    sim.reset_meep()

    geometry = [mp.Block(mp.Vector3(w1, custom, h),
                    center=mp.Vector3(x1 - sz/2, 0, offset),
                    material=mp.Medium(epsilon=epsi_Si)),
            mp.Block(mp.Vector3(w2, custom, h),
                    center=mp.Vector3(x2 - sz/2, 0, offset),
                    material=mp.Medium(epsilon=epsi_Si)),
            mp.Block(mp.Vector3(mp.inf, custom, h),
                    center=mp.Vector3(0, 0, offset + h),
                    material=mp.Medium(epsilon=epsi_SiO2))]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        dimensions=dimensions,
        resolution=resolution,
    )

    plt.figure(dpi=100)
    sim.plot2D()
    plt.savefig('visual')

    sim.plot2D(fields=mp.Ex)
    plt.savefig('visual_field')
    
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    sim.load_minus_flux_data(refl, empty_data)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(z=-0.5 * sz + dpml), 1e-9
        )
    )

    refl_flux = mp.get_fluxes(refl)
    freqs = mp.get_flux_freqs(refl)

    wvls = np.empty(nfreq)
    theta_out = np.empty(nfreq)
    R = np.empty(nfreq)
    for i in range(nfreq):
        wvls[i] = 1 / freqs[i]
        theta_out[i] = math.degrees(math.asin(k.x / freqs[i]))
        R[i] = -refl_flux[i] / empty_flux[i]
        print("refl:, {}, {}, {}, {}, {}".format(k.x, wvls[i], theta, theta_out[i], R[i]))

    return k.x * np.ones(nfreq), wvls, theta_out, R

theta_in = np.arange(-20,21,5)
wvl = np.empty(nfreq)
kxs = np.empty((nfreq, theta_in.size))
thetas = np.empty((nfreq, theta_in.size))
Rmeep = np.empty((nfreq, theta_in.size))

for j in range(theta_in.size):
    kxs[:, j], wvl, thetas[:, j], Rmeep[:, j] = planar_reflectance(theta_in[j])

# create a 2d matrix for the wavelength by repeating the column vector for each angle
wvls = np.transpose(np.matlib.repmat(wvl, theta_in.size, 1))

plt.plot(theta_in, Rmeep[int(nfreq/1.5)], label = 'meep')
plt.xlabel('thetas')
plt.ylabel('reflectance')
plt.savefig('attempt1')