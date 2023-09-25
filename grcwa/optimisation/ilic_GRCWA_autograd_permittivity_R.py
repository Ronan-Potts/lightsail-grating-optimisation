""" Topology optimization of reflection of a single patterned layers ."""
""" Nlopt is needed. For some reason does not work with numpy 1.25.2. Works fine with numpy 1.25.0 """

import grcwa
grcwa.set_backend('autograd')  # important!!

import numpy as np
import autograd.numpy as npgrad
import matplotlib.pyplot as plt
from autograd import grad

try:
    import nlopt
    NL_AVAILABLE = True
except ImportError:
    NL_AVAILABLE = False

if NL_AVAILABLE == False:
    raise Exception('Please install NLOPT')

'''
Discretisation values
'''
# Truncation order
nG = 30
# Resolution 
Nx = 30
Ny = 1


obj_order = 0  # <----- WHICH ORDER DO YOU WANT TO LOOK AT?

'''
Lattice cell parameters
'''
# Defining Ilic dimensions
d = 1.8  # unit cell width
dy = 1e-1
x1 = 0.85*d # positions of blocks in unit cell
x2 = 0.25*d 
w1 = 0.15*d # width of blocks in unit cell
w2 = 0.35*d

h = 0.5  # thickness of resonator layer
t = 0.5  # also equal to thickness of substrate layer

E_vacuum = 1.
E_Si = 3.5**2    # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)
E_SiO2 = 1.45**2  # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)

# Lattice constants. I will consider a lattice with square-shaped unit cells of size d x d (in natural units specified above)
L1 = [d, 0]
L2 = [0, dy]

'''
Light approach
'''
wavelength = 1.5
freq = 1./wavelength
# INPUT ANGLES IN DEGREES
theta = 0.
phi = 0.

# planewave excitation
planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}


'''
Cost function for optimisation.

       x:   The dielectric constant on the 2D grids of size Nx*Ny
       
    Qabs:   A parameter for relaxation to better approach global optimal, at Qabs = inf, it will describe the real physics.
            It also be used to resolve the singular matrix issues by setting a large but finite Qabs, e.g. Qabs = 1e5
'''

def cost_function(x,Qabs):
    freqcmp = freq*(1+1j/2/Qabs)
    ######### setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta*np.pi/180,phi*np.pi/180,verbose=0)
    # input layer information
    obj.Add_LayerUniform(0,E_vacuum)     # Layer 0
    obj.Add_LayerGrid(h,Nx,Ny)           # Layer 1
    obj.Add_LayerUniform(t,E_SiO2)       # Layer 2
    obj.Add_LayerUniform(0,E_vacuum)     # Layer 3
    '''
                                    _________
                                     Layer 0 (vacuum)
       _____     ____               _________
      |     |   |    |               Layer 1 (pattern)
    __|     |___|    |__            _________
    |                    |   ...     Layer 2 (surface)
    |____________________|          _________
                                     Layer 3 (vacuum)
                                    _________
    '''
    obj.Init_Setup()

    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
    obj.GridLayer_geteps(x)
    # compute reflection and transmission
    R,T= obj.RT_Solve(normalize=1)
    return R

# For animated figure ##############
plt.ion()
fig1, ax1 = plt.subplots()
####################################

# nlopt function
ctrl = 0
Qabs = npgrad.inf
fun = lambda x: cost_function(x,Qabs)
grad_fun = grad(fun)
def fun_nlopt(x,gradn):
    global ctrl
    gradn[:] = grad_fun(x)
    y = fun(x)
    
    print('Step = ',ctrl,', R = ',y)
    # Visualising the geometry ____________________________________________________________________
    if ctrl == 0:
        global anim1
        anim1 = ax1.imshow(np.reshape(x, (Nx,Ny)), interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
        cbar = plt.colorbar(anim1)
        cbar.set_label("Permittivity")
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title(r"Step {}, $R = {}$".format(ctrl, round(y,5)))
        plt.savefig('ISB B/grcwa/optimisation/figs/permittivity/ilic_GRCWA_optimisation_permittivity_R_initial')
    else:
        anim1.set_data(np.reshape(x, (Nx,Ny)))
        plt.title(r"Step {}, $R = {}$".format(ctrl, round(y,5)))
        fig1.canvas.flush_events()
    # _____________________________________________________________________________________________
    ctrl += 1
    return fun(x)


'''
Cell geometry
'''
# Unit cell geometry (rows, cols)
cell_geometry = np.ones((Nx,Ny), dtype=float)*E_vacuum

x0 = np.linspace(0,d,Nx)
y0 = np.linspace(0,dy,Ny)
x, y = np.meshgrid(x0,y0, indexing='ij')
# The design will be completely uniform in the y-direction.
filter = (abs(x-x1) <= w1/2) | (abs(x-x2) <= w2/2)
cell_geometry[filter] = E_Si


'''
NLOPT Setup
'''
# set up NLOPT
ndof = Nx*Ny
init = cell_geometry.flatten()
lb=npgrad.ones(ndof,dtype=float)*E_vacuum
ub=npgrad.ones(ndof,dtype=float)*E_Si

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(100)

opt.set_max_objective(fun_nlopt)
x = opt.optimize(init)




plt.imshow(np.reshape(x, (Nx,Ny)), interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
plt.xlabel("y")
plt.ylabel("x")
plt.title(r"Final result, Step {}, $R_0 = {}$".format(ctrl, round(fun(x),5)))
plt.savefig('ISB B/grcwa/optimisation/figs/permittivity/ilic_GRCWA_optimisation_permittivity_R_final')