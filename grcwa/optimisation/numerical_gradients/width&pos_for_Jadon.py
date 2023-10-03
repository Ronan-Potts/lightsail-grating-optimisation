""" Topology optimization of reflection of a single patterned layer."""
""" Nlopt is needed. For some reason does not work with latest version of numpy. Works fine with numpy 1.25.0 """

import grcwa
grcwa.set_backend('autograd')  # important!!

import numpy as np
import autograd.numpy as npgrad
# from autograd import grad
import nlopt

import matplotlib.pyplot as plt

'''
Discretisation values
'''
# Truncation order
nG = 30
# Resolution 
Nx = 10000
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
E_Si = 3.5**2    # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)
E_SiO2 = 1.45**2  # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)

# Lattice constants. I will consider a lattice with square-shaped unit cells of size d x d (in natural units specified above)
L1 = [d, 0]
L2 = [0, dy]

'''
Cell geometry
    vars: an array, vars = [x1,w1,x2,w2]
'''
def get_cell_geometry(vars):
    # Unit cell geometry (rows, cols)
    x = np.ones((Nx,Ny), dtype=float)*E_vacuum

    x0 = np.linspace(0,d,Nx)
    y0 = np.linspace(0,dy,Ny)
    x, y = np.meshgrid(x0,y0, indexing='ij')
    # The design will be completely uniform in the y-direction.
    filter = (abs(x-vars[0]) <= vars[1]/2) | (abs(x-vars[2]) <= vars[3]/2)
    x[filter] = E_Si
    return x

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

       vars:   [x1,w1,x2,w2]
       
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
    return 2*e0 + (e1 + en1)*(1 + np.sqrt( 1 - (wavelength/d)**2 ))

# For animated figure ##############
plt.ion()
fig1, ax1 = plt.subplots()
####################################

# nlopt function
ctrl = 0
Qabs = npgrad.inf
fun = lambda vars: cost_function(vars,Qabs)
# grad_fun = grad(fun)
def fun_nlopt(vars,gradn):
    global ctrl
    # gradn[:] = grad_fun(vars)
    '''
    autoGrad doesn't work well here as space is discretised. Instead I will numerically evaluate the gradient.
    '''
    delx1_fun = fun([vars[0]+d/Nx, vars[1],      vars[2],      vars[3]])      - fun(vars)
    delw1_fun = fun([vars[0],      vars[1]+d/Nx, vars[2],      vars[3]])      - fun(vars)
    delx2_fun = fun([vars[0],      vars[1],      vars[2]+d/Nx, vars[3]])      - fun(vars)
    delw2_fun = fun([vars[0],      vars[1],      vars[2],      vars[3]+d/Nx]) - fun(vars)
    gradn[:] = [delx1_fun, delw1_fun, delx2_fun, delw2_fun]
    y = fun(vars)

    # Printing parameters to command line
    if ctrl == 0:
        print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format("Step", "Term 2", 'x1', 'w1', 'x2', 'w2'))
    else:
        print("{:<8} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f}".format(ctrl, y, vars[0],vars[1],vars[2],vars[3]))
    # Visualising the geometry ____________________________________________________________________
    if ctrl == 0:
        global anim1
        anim1 = ax1.imshow(get_cell_geometry(vars), interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
        cbar = plt.colorbar(anim1)
        cbar.set_label("Permittivity")
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title("Step {}, Term 2 = {}".format(ctrl, round(y,5)))
        plt.savefig('grcwa/optimisation/numerical_gradients/figs/numerical_width&pos_for_Jadon_initial')

    else:
        anim1.set_data(get_cell_geometry(vars))
        plt.title("Step {}, Term 2 = {}".format(ctrl, round(y,5)))
        fig1.canvas.flush_events()
    # _____________________________________________________________________________________________
    ctrl += 1
    return y




'''
NLOPT Setup
'''
# set up NLOPT
ndof = 4
init = [x1,w1,x2,w2]
lb = [0,0,0,0]
ub = [d,d,d,d]

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
plt.title("Final result, Step {}, Term 2 = {}".format(ctrl, round(fun(vars),5)))
plt.savefig('grcwa/optimisation/numerical_gradients/figs/numerical_width&pos_for_Jadon_final')