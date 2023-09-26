""" Topology optimization of reflection of a single patterned layers ."""
""" Nlopt is needed. For some reason does not work with latest version of numpy. Works fine with numpy 1.25.0 """

import grcwa
grcwa.set_backend('autograd')  # important!!

import numpy as np
import autograd.numpy as npgrad
import nlopt
from autograd import grad

import matplotlib.pyplot as plt

'''
Discretisation values
'''
# Truncation order
nG = 30
# Resolution 
Nx = 100
Ny = 1

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
E_Si = 3.5**2     # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)
E_SiO2 = 1.45**2  # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)

# Lattice constants. I will consider a lattice with square-shaped unit cells of size d x d (in natural units specified above)
L1 = [d, 0]
L2 = [0, dy]

def boundary_indices(x):
    global E_Si
    # Define boundary permittivities
    eps_1d = x[:,0]
    boundary_indices = []
    for i in range(1,len(eps_1d)-1):
        lower_neighbor = eps_1d[i-1]
        current = eps_1d[i]
        upper_neighbor = eps_1d[i+1]
        if current == E_Si and (lower_neighbor == E_Si or upper_neighbor == E_Si) and (lower_neighbor != upper_neighbor):
            # then we are at a boundary
            boundary_indices.append(i)
    return boundary_indices
'''
Cell geometry
    vars: an array, vars = [eps1, eps2] which contains the permittivities at the edges of each element.
'''
def get_cell_geometry(vars):
    
    # Unit cell geometry (rows, cols)
    cell_geometry = np.ones((Nx,Ny), dtype=float)*E_vacuum

    x0 = np.linspace(0,d,Nx)
    y0 = np.linspace(0,dy,Ny)
    x, y = np.meshgrid(x0,y0, indexing='ij')

    # The design will be completely uniform in the y-direction.
    filter = (abs(x-x1) <= vars[0]/2) | (abs(x-x2) <= vars[1]/2)
    cell_geometry[filter] = E_Si

    # What proportion of the width overhangs into a slice?
    prop_w1 = (vars[0]/(2*Nx)) % 1
    prop_w2 = (vars[1]/(2*Nx)) % 1

    eps_w1 = 1 + 11*prop_w1
    eps_w2 = 1 + 11*prop_w2

    boundary_i = boundary_indices(cell_geometry)
    lower_boundary = boundary_i[0:2]
    upper_boundary = boundary_i[2:]
    
    cell_geometry[lower_boundary,:] = eps_w1
    cell_geometry[upper_boundary,:] = eps_w2

    return cell_geometry

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

def cost_function(vars,Qabs):
    # Cell geometry
    cell_geometry = get_cell_geometry(vars)
    '''
    Building layers
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
    freqcmp = freq*(1+1j/2/Qabs)
    ######### setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta*np.pi/180,phi*np.pi/180,verbose=0)
    # input layer information
    obj.Add_LayerUniform(0,E_vacuum)     # Layer 0
    obj.Add_LayerGrid(h,Nx,Ny)           # Layer 1
    obj.Add_LayerUniform(t,E_SiO2)       # Layer 2
    obj.Add_LayerUniform(0,E_vacuum)     # Layer 3
    obj.Init_Setup()


    '''
    Solving Maxwell's equations
    '''
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
    obj.GridLayer_geteps(cell_geometry.flatten())
    # compute reflection and transmission by order
    # Ri(Ti) has length obj.nG, to see which order, check obj.G; too see which kx,ky, check obj.kx obj.ky
    R,T= obj.RT_Solve(normalize=1)
    return R

# For animated figure ##############
plt.ion()
fig1, ax1 = plt.subplots()
####################################

# nlopt function
ctrl = 0
Qabs = npgrad.inf
fun = lambda vars: cost_function(vars,Qabs)
grad_fun = grad(fun)
def fun_nlopt(vars,gradn):
    global ctrl
    # AVM: dC/dw = 1/(e-1)  *  dC/de
    gradn[:] = grad_fun(vars)
    y = fun(vars)

    # Printing parameters to command line
    if ctrl == 0:
        print("{:<8} {:<8} {:<8} {:<8}".format("Step", "R", 'w1', 'w2'))
    else:
        print("{:<8} {:<8.3f} {:<8.3f} {:<8.3f}".format(ctrl, y, vars[0],vars[1]))
    # Visualising the geometry ____________________________________________________________________
    if ctrl == 0:
        global anim1
        anim1 = ax1.imshow(get_cell_geometry(vars), interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
        cbar = plt.colorbar(anim1)
        cbar.set_label("Permittivity")
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title(r"Step {}, $R = {}$".format(ctrl, round(y,5)))
        plt.savefig('grcwa/optimisation/figs/width/ilic_GRCWA_optimisation_width_R_initial')

    else:
        anim1.set_data(get_cell_geometry(vars))
        plt.title(r"Step {}, $R = {}$".format(ctrl, round(y,5)))
        fig1.canvas.flush_events()
    # _____________________________________________________________________________________________
    ctrl += 1
    return y




'''
NLOPT Setup
'''
# set up NLOPT
ndof = 2
init = [w1,w2]
lb = [0,0]
ub = [d,d]

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(100)

opt.set_max_objective(fun_nlopt)
vars = opt.optimize(init)


plt.imshow(get_cell_geometry(vars), interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
plt.xlabel("y")
plt.ylabel("x")
plt.title(r"Final result, Step {}, $R = {}$".format(ctrl, round(fun(vars),5)))
plt.savefig('grcwa/optimisation/figs/width/ilic_GRCWA_optimisation_width_R_final')
'''
PROBLEMS:

 1) gradn[:] is zero. Not sure why. Is likely due to small changes in 'vars' which cause no change in get_cell_geometry due to discretisation.
 2) I can't decrease Nx by too much - it is limited by the size of d, where larger d requires larger Nx for indexing purposes. GRCWA seems to
    discretise in fixed amounts at some stage (not very good). This is likely the cause of the problem (1).
'''