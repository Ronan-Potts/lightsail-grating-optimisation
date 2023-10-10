""" Topology optimization of reflection of a single patterned layers ."""
""" Nlopt is needed. For some reason does not work with latest version of numpy. Works fine with numpy 1.25.0 """

import grcwa
grcwa.set_backend('autograd')  # important!!

import numpy as numpy
import autograd.numpy as np
import nlopt
from autograd import grad

import matplotlib.pyplot as plt

'''
Discretisation values
'''
# Truncation order
nG = 30
# Resolution 
Nx = 161
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

'''
Light approach
'''
wavelength = 1.5
freq = 1./wavelength
# INPUT ANGLES IN DEGREES
theta = 0.
phi = 0.

# Speed of light
c = 1

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
def En1_func(theta, cell_geometry):
    # It is necessary to ensure that cell_geometry is not an ArrayBox, which does happen whenever grad_fun is called.
    if type(cell_geometry) == np.numpy_boxes.ArrayBox:
        cell_geometry = cell_geometry._value
    
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
    # Ri(Ti) has length obj.nG, to see which order, check obj.G; to see which kx,ky, check obj.kx obj.ky
    Ri,Ti= obj.RT_Solve(byorder=1)
    ords = obj.G # Returns a list of tuples [ord1, ord2] where ord1 is the order in the L1 direction, while ord2 is the order in the L2 direction.

    # Compute the reflectance and transmittance for a particular theta at various orders
    Rn1 = sum(Ri[ords[:,0] == -1])
    Tn1 = sum(Ti[ords[:,0] == -1])
    en1 = (Rn1 + Tn1)/(sum(Ri) + sum(Ti))
    return en1
    

def cost_function(vars,Qabs,v):
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
    # Ri(Ti) has length obj.nG, to see which order, check obj.G; to see which kx,ky, check obj.kx obj.ky
    Ri,Ti= obj.RT_Solve(byorder=1)
    ords = obj.G # Returns a list of tuples [ord1, ord2] where ord1 is the order in the L1 direction, while ord2 is the order in the L2 direction.

    # Compute the reflectance and transmittance for a particular theta at various orders
    R0 = sum(Ri[ords[:,0] == 0])
    R1 = sum(Ri[ords[:,0] == 1])
    Rn1 = sum(Ri[ords[:,0] == -1])
    T0 = sum(Ti[ords[:,0] == 0])
    T1 = sum(Ti[ords[:,0] == 1])
    Tn1 = sum(Ti[ords[:,0] == -1])
    e0 = (R0 + T0)/(sum(Ri) + sum(Ti))
    e1 = (R1 + T1)/(sum(Ri) + sum(Ti))
    en1 = (Rn1 + Tn1)/(sum(Ri) + sum(Ti))

    # Finding pEn1_pTheta
    En1_fun = lambda theta: En1_func(theta, cell_geometry)
    pEn1_pTheta = grad(En1_fun)(0.)

    # Relativistic terms
    c = 1
    beta = v/c
    gamma = 1/np.sqrt(1-beta**2)
    D1 = beta*gamma + gamma - 1

    return -(1/v)*(D1**2)*( 2*(wavelength/d)*pEn1_pTheta*(1/D1 - 1) - (gamma-1)*(2*e0 + (e1 + en1)*(1 + np.sqrt( 1 - (wavelength/d)**2 ))) )

# For animated figure ##############
plt.ion()
fig1, ax1 = plt.subplots()
####################################

# nlopt function
ctrl = 0
Qabs = np.inf
v = 0.2*c
fun = lambda vars: cost_function(vars,Qabs,v)
grad_fun = grad(fun)

def fun_nlopt(vars,gradn):
    global ctrl
    
    gradn[:] = grad_fun(vars)
    y = fun(vars)

    # Printing parameters to command line
    if ctrl == 0:
        print("{:<8} {:<20} {:<8} {:<8}".format("Step", "Force Component", 'w1', 'w2'))
        print("{:<8} {:<20.3f} {:<8.3f} {:<8.3f}".format(ctrl, y, vars[0],vars[1]))

        
        global anim1
        anim1 = ax1.imshow(get_cell_geometry(vars,Nx,Ny,d,dy,x1,x2), interpolation='nearest', vmin=E_vacuum, vmax=E_Si, aspect='auto')
        cbar = plt.colorbar(anim1)
        cbar.set_label("Permittivity")
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title(r"Step {}, Force Component = {} $I A' v_y / c$".format(ctrl, round(y,5)))
        plt.savefig('grcwa/optimisation/figs/nlopt_width_for_force_initial')
    else:
        print("{:<8} {:<20.3f} {:<8.3f} {:<8.3f}".format(ctrl, y, vars[0],vars[1]))

        anim1.set_data(get_cell_geometry(vars,Nx,Ny,d,dy,x1,x2))
        plt.title(r"Step {}, Force Component = {} $I A' v_y / c$".format(ctrl, round(y,5)))
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
ub = [0.3*d,0.5*d]

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(100)

opt.set_min_objective(fun_nlopt)
vars = opt.optimize(init)


plt.imshow(get_cell_geometry(vars,Nx,Ny,d,dy,x1,x2), interpolation='nearest', vmin=E_vacuum, vmax=E_Si, aspect='auto')
plt.xlabel("y")
plt.ylabel("x")
plt.title(r"Final result, Step {}, Force Component = {} $I A' v_y / c$".format(ctrl, round(fun(vars),5)))
plt.savefig('grcwa/optimisation/figs/nlopt_width_for_force_final')