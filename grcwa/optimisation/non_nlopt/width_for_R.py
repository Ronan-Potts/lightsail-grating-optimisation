import grcwa
grcwa.set_backend('autograd')  # important!!

import numpy as numpy
import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt

'''
Discretisation values
'''
# Truncation order
nG = 30
# Resolution 
Nx = 41
Ny = 1


'''
Optimisation parameters
'''
max_iterations = 1000
step_size = 0.001

precision = 10       # rounding gradient to 10^-precision before checking whether gradient is zero

decimal_precision = 3 # rounding the cost function before checking equivalence with previous step's cost
num_identical=10      # for how any steps should the cost function be identical?

'''
Lattice cell parameters
'''
# Defining Ilic dimensions
d = 1.8  # unit cell width
dy = 1e-1
x1 = 0.25*d # positions of blocks in unit cell
x2 = 0.85*d 
w1 = 0.35*d # width of blocks in unit cell
w2 = 0.15*d

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
Finding w_filled
'''

def w_filled(Nx,Ny,d,dy,x1,x2,w1,w2):
    ones = np.ones((Nx,Ny))

    # Defining coordinate system
    x0 = np.linspace(0,d,Nx)
    y0 = np.linspace(0,dy,Ny)
    x, y = np.meshgrid(x0,y0, indexing='ij')

    filter1 = abs(x - x1) <= w1/2
    filter2 = abs(x-x2) <= w2/2
    cell_geometry = np.ones((Nx,Ny)) * E_vacuum
    cell_geometry[filter1 | filter2] = E_Si
    # How many cells are filled?
    w_filled_e1 = sum(ones[filter1]*d/(Nx-1))
    w_filled_e2 = sum(ones[filter2]*d/(Nx-1))

    # Calculate boundary permittivities using width proportion unaccounted for
    eps1 = (E_Si - E_vacuum)*(w1-w_filled_e1)/(2*d/(Nx-1)) + E_vacuum
    eps2 = (E_Si - E_vacuum)*(w2-w_filled_e2)/(2*d/(Nx-1)) + E_vacuum

    # Find boundary indices
    cell_geom_1d = cell_geometry[:,0]
    boundary_indices = []
    for i in range(1,len(cell_geom_1d)-1):
        lower_neighbor = cell_geom_1d[i-1]
        current = cell_geom_1d[i]
        upper_neighbor = cell_geom_1d[i+1]
        if (current <= E_Si and current > E_vacuum) and (lower_neighbor == E_Si or upper_neighbor == E_Si) and (lower_neighbor != upper_neighbor):
            # then we are at a boundary
            boundary_indices.append(i)

    return cell_geometry, eps1, eps2, boundary_indices


'''
Now get the cell geometry with boundary permittivities adjusted for width
'''

def get_cell_geometry(vars,Nx,Ny,d,dy,x1,x2):
    w1 = vars[0]
    w2 = vars[1]

    # The design will be completely uniform in the y-direction.
    cell_geometry, eps1, eps2, boundary_indices = w_filled(Nx,Ny,d,dy,x1,x2,w1,w2)
    lower_boundary = boundary_indices[0:2]
    upper_boundary = boundary_indices[2:]
    
    # Change boundary permittivities
    cell_geometry = cell_geometry.tolist()
    cell_geometry[lower_boundary[0]][0] = eps1
    cell_geometry[lower_boundary[1]][0] = eps1
    cell_geometry[upper_boundary[0]][0] = eps2
    cell_geometry[upper_boundary[1]][0] = eps2
    cell_geometry = np.array(cell_geometry)
    
    return cell_geometry







'''
Cost function for optimisation.

       x:   The dielectric constant on the 2D grids of size Nx*Ny
       
    Qabs:   A parameter for relaxation to better approach global optimal, at Qabs = inf, it will describe the real physics.
            It also be used to resolve the singular matrix issues by setting a large but finite Qabs, e.g. Qabs = 1e5
'''

def cost_function(vars,Qabs):
    # Cell geometry
    cell_geometry = get_cell_geometry(vars,Nx,Ny,d,dy,x1,x2)
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


Qabs = np.inf

cost_fun = lambda vars: cost_function(vars,Qabs)
grad_fun = grad(cost_fun)

def fun_grad(vars):
    global grad
    grad = grad_fun(vars)
    cost_val = cost_fun(vars)

    return cost_val

'''
Optimisation Setup
'''

# Initialise
vars = [w1, w2]

fun_grad(vars)
index = 0
outputs = []

# For animated figure ##############
plt.ion()
fig1, ax1 = plt.subplots()
####################################

# Define loop to move in direction of grad until maximum is reached
while index <= max_iterations:
    cell_geometry = get_cell_geometry(vars,Nx,Ny,d,dy,x1,x2)
    old_grad = grad
    cost_val = fun_grad(vars)
    outputs.append(cost_val)

    # Printing parameters to command line
    if index == 0:
        print("{:<8} {:<8} {:<8} {:<8}".format("Step", "R", 'w1', 'w2'))
        print("{:<8} {:<8.3f} {:<8.3f} {:<8.3f}".format(index, cost_val, vars[0],vars[1]))

        anim1 = ax1.imshow(cell_geometry, interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
        cbar = plt.colorbar(anim1)
        cbar.set_label("Permittivity")
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title(r"Step {}, $R = {:.5f}$".format(index, round(cost_val,decimal_precision)))
        plt.savefig('grcwa/optimisation/figs/width_for_R_initial')
    else:
        print("{:<8} {:<8.3f} {:<8.3f} {:<8.3f}".format(index, cost_val, vars[0],vars[1]))

        anim1.set_data(cell_geometry)
        plt.title(r"Step {}, $R = {:.5f}$".format(index, round(cost_val,decimal_precision)))
        fig1.canvas.flush_events()


    if (round(float(grad[0]),precision) == 0 and round(float(grad[1]),precision) == 0) or (index >= 10 and len(set(np.round(np.array(outputs[-1*num_identical:]), decimal_precision)))==1):
        print("Optimum reached after {} steps. Terminating the loop...".format(index))
        break

    
    
    old_cost = cost_val
    old_vars = vars
    # Change vars using grad
    vars = vars + step_size*np.array(grad)
    cost_val = fun_grad(vars)

    if old_cost > cost_val:
        step_size = 0.9*step_size
    
    index += 1

if index == max_iterations + 1:
    print("Solution could not converge after {} steps".format(index-1))

plt.imshow(get_cell_geometry(vars,Nx,Ny,d,dy,x1,x2), interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
plt.xlabel("y")
plt.ylabel("x")
plt.title(r"Final result, Step {}, $R = {:.5f}$".format(index, round(cost_fun(vars),decimal_precision)))
plt.savefig('grcwa/optimisation/figs/width_for_R_final')

