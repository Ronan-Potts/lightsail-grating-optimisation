import grcwa
grcwa.set_backend('autograd')  # important!!

import numpy as numpy
import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt


E_vacuum = 1.
E_Si = 3.5**2    # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)
E_SiO2 = 1.45**2  # https://refractiveindex.info/ with k = n^2 (dielectric constant, refractive index)

Ny = 1
dy = 1e-4

# planewave excitation
planewave = {'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
phi = 0.

## Unit cell dimension
L2 = [0,dy]

## Thickness parameters
h = 0.5  # thickness of resonator layer
t = 0.5  # also equal to thickness of substrate layer
v_width = 2    # thickness of vacuum layers

## Sail speed
c = 1.
v = 0.2*c
def linear_boundary_geometry(Nx,d,x1,x2,w1,w2):
    '''
    This function defines a 2D two-element unit cell with the widths and positions of the elements given.

        Nx: the number of slices used to spatially discretise the unit cell.
        
        d: the size of the unit cell

        x1: the central position of element 1 in the unit cell

        x2: the central position of element 2 in the unit cell

        w1: the width of element 1 in the unit cell

        w2: the width of element 2 in the unit cell
    '''
    if x1/(d/(Nx-1)) - int(x1/(d/(Nx-1))) > 0:
        raise Exception("x1 must be an integer multiple of the slice width d/(Nx-1).")
        # ... otherwise I would need to make the code needlessly complicated.
    if x2/(d/(Nx-1)) - int(x2/(d/(Nx-1))) > 0:
        raise Exception("x2 must be an integer multiple of the slice width d/(Nx-1).")
        # ... otherwise I would need to make the code needlessly complicated.
    

    ### Generating basic unit cell
    ones = np.ones((Nx,Ny))

    # Defining coordinate system
    x0 = np.linspace(0,d,Nx)
    y0 = np.linspace(0,dy,Ny)
    x, y = np.meshgrid(x0,y0, indexing='ij')

    filter1 = abs(x - x1) <= w1/2
    filter2 = abs(x-x2) <= w2/2
    cell_geometry = np.ones((Nx,Ny)) * E_vacuum
    cell_geometry[filter1 | filter2] = E_Si

    ### Correcting boundary permittivities for continuity of width in unit cell
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

    lower_boundary = boundary_indices[0:2]
    upper_boundary = boundary_indices[2:]

    # Change boundary permittivities
    cell_geometry = cell_geometry.tolist()
    cell_geometry[lower_boundary[0]][0] = eps1
    cell_geometry[lower_boundary[1]][0] = eps1
    cell_geometry[upper_boundary[0]][0] = eps2
    cell_geometry[upper_boundary[1]][0] = eps2
    cell_geometry = np.array(cell_geometry)

    # cell_geometry is a 1D array whose values correspond to the permittivity of each spatial slice in the unit cell.
    # eps1 and eps2 are the permittivites at the boundaries of elements 1 and 2 respectively.
    return cell_geometry, eps1, eps2


def linear_permittivity_geometry(Nx,d,x1,x2,w1,w2,eps1,eps2):
    '''
    This function defines a 2D two-element unit cell with the widths and positions of the elements given, but the boundary
    permittivity of the unit cell specified externally.

        Nx: the number of slices used to spatially discretise the unit cell.
        
        d: the size of the unit cell

        x1: the central position of element 1 in the unit cell

        x2: the central position of element 2 in the unit cell

        w1: the width of element 1 in the unit cell

        w2: the width of element 2 in the unit cell

        eps1: the permittivity at the boundary of element 1

        eps2: the permittivity at the boundary of element 2
    '''

    if w1/(2*d/(Nx-1)) - int(w1/(2*d/(Nx-1))) > 0:
        raise Exception("w1 must be an even integer multiple of the slice width d/(Nx-1).")
        # ... otherwise I would need to make the code needlessly complicated.
    if w2/(2*d/(Nx-1)) - int(w2/(2*d/(Nx-1))) > 0:
        raise Exception("w2 must be an even integer multiple of the slice width d/(Nx-1).")
        # ... otherwise I would need to make the code needlessly complicated.
    if x1 > x2:
        raise Exception("x1 should be less than x2, otherwise eps1 will correspond to element 2.")


    ### Finding actual_w1 and actual_w2
    prop_w1 = 1 - (eps1 - E_vacuum)/(E_Si - E_vacuum)
    prop_w2 = 1 - (eps2 - E_vacuum)/(E_Si - E_vacuum)
    
    actual_w1 = w1 - (2*d/(Nx-1))*prop_w1
    actual_w2 = w2 - (2*d/(Nx-1))*prop_w2
    
    ### Generating basic unit cell
    # Defining coordinate system
    x0 = np.linspace(0,d,Nx)
    y0 = np.linspace(0,dy,Ny)
    x, y = np.meshgrid(x0,y0, indexing='ij')

    filter1 = abs(x - x1) <= w1/2
    filter2 = abs(x-x2) <= w2/2
    cell_geometry = np.ones((Nx,Ny)) * E_vacuum
    cell_geometry[filter1 | filter2] = E_Si

    ### Finding boundaries using the basic unit cell with eps = E_Si or E_vacuum
    eps_1d = cell_geometry.flatten()

    boundary_indices = []
    for i in range(1,len(eps_1d)-1):
        lower_neighbor = eps_1d[i-1]
        current = eps_1d[i]
        upper_neighbor = eps_1d[i+1]
        if current == E_Si and (lower_neighbor == E_Si or upper_neighbor == E_Si) and (lower_neighbor != upper_neighbor):
            # then we are at a boundary
            boundary_indices.append(i)

    # Lower boundary is first two values, upper boundary is later two values
    lower_boundary = boundary_indices[0:2]
    upper_boundary = boundary_indices[2:]

    ### Change boundary permittivities
    cell_geometry = cell_geometry.tolist()     # if I don't convert to list, I get a weird numpy value error.
    cell_geometry[lower_boundary[0]][0] = eps1
    cell_geometry[lower_boundary[1]][0] = eps1
    cell_geometry[upper_boundary[0]][0] = eps2
    cell_geometry[upper_boundary[1]][0] = eps2
    cell_geometry = np.array(cell_geometry)
    

    # cell_geometry is a 1D array whose values correspond to the permittivity of each spatial slice in the unit cell.
    # actual_w1 and actual_w2 are the proper values of w1 and w2 by using a linear model which equates a partially filled
    # slice in the unit cell with a permittivity in the range [E_vacuum,E_Si].
    return cell_geometry, actual_w1, actual_w2
    
    
def grcwa_reflectance_transmittance(nG,cell_geometry,Nx,d,theta,freq):
    '''
    Calculates the reflectance of the infinitely extended unit cell.

        nG: truncation order for spatial discretisation. Affects grating diffraction orders.

        cell_geometry: the geometry of the unit cell, described as either a 1D or 2D array of permittivities.

        Nx: the number of slices used to spatially discretise the unit cell.

        d: width of the unit cell.

        theta: incident angle of light.

    '''
    ## Frequency
    Qabs = np.inf
    freqcmp = freq*(1+1j/2/Qabs)

    ## Unit cell dimension
    L1 = [d,0]

    ## setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta*np.pi/180,phi*np.pi/180,verbose=0)

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
    # input layer information
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 0
    obj.Add_LayerGrid(h,Nx,Ny)           # Layer 1
    obj.Add_LayerUniform(t,E_SiO2)       # Layer 2
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 3
    obj.Init_Setup()

    ## Input plane wave and grid information
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
    obj.GridLayer_geteps(cell_geometry.flatten())

    ## Compute reflectance and transmittance
    R,T= obj.RT_Solve(normalize=1)
    return R,T

def grcwa_reflectance_transmittance_orders(nG,orders,cell_geometry,Nx,d,theta,freq):
    '''
    Calculates the reflectance of the infinitely extended unit cell.

        nG: truncation order for spatial discretisation. Affects grating diffraction orders.

        cell_geometry: the geometry of the unit cell, described as either a 1D or 2D array of permittivities.

        Nx: the number of slices used to spatially discretise the unit cell.

        d: width of the unit cell.

        theta: incident angle of light.

    '''
    ## Frequency
    Qabs = np.inf
    freqcmp = freq*(1+1j/2/Qabs)

    ## Unit cell dimension
    L1 = [d,0]

    ## setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta*np.pi/180,phi*np.pi/180,verbose=0)

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
    # input layer information
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 0
    obj.Add_LayerGrid(h,Nx,Ny)           # Layer 1
    obj.Add_LayerUniform(t,E_SiO2)       # Layer 2
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 3
    obj.Init_Setup()

    ## Input plane wave and grid information
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
    obj.GridLayer_geteps(cell_geometry.flatten())

    ## Compute reflectance and transmittance by order
    # Ri(Ti) has length obj.nG, to see which order, check obj.G; too see which kx,ky, check obj.kx obj.ky
    Ri,Ti= obj.RT_Solve(byorder=1)
    ords = obj.G # Returns a list of tuples [ord1, ord2] where ord1 is the order in the L1 direction, while ord2 is the order in the L2 direction.
    # Compute the reflectance and transmittance for a particular theta at various orders
    Rs = np.array([])
    Ts = np.array([])
    for i in orders:
        # Calculate reflection and transmission at angle=theta for various orders i=0,1,-1
        Rs = np.append(Rs, sum(Ri[ords[:,0] == i]))
        Ts = np.append(Ts, sum(Ti[ords[:,0] == i]))
        
    return Rs,Ts



def grcwa_efficiencies(nG,orders,cell_geometry,Nx,d,theta,freq):
    '''
    Calculates the diffraction order efficiencies for the infinitely extended unit cell.

        nG: truncation order for spatial discretisation. Affects grating diffraction orders.

        cell_geometry: the geometry of the unit cell, described as either a 1D or 2D array of permittivities.

        Nx: the number of slices used to spatially discretise the unit cell.

        d: width of the unit cell.

        theta: incident angle of light.
    '''
    ## Frequency
    Qabs = np.inf
    freqcmp = freq*(1+1j/2/Qabs)

    ## Unit cell dimension
    L1 = [d,0]

    # It is necessary to ensure that cell_geometry is not an ArrayBox, which does happen whenever grad_fun is called.
    if type(cell_geometry) == np.numpy_boxes.ArrayBox:
        cell_geometry = cell_geometry._value

    ## setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta*np.pi/180,phi*np.pi/180,verbose=0)

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
    # input layer information
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 0
    obj.Add_LayerGrid(h,Nx,Ny)           # Layer 1
    obj.Add_LayerUniform(t,E_SiO2)       # Layer 2
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 3
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
    Rs = np.array([])
    Ts = np.array([])
    for i in orders:
        # Calculate reflection and transmission at angle=theta for various orders i=0,1,-1
        Rs = np.append(Rs, sum(Ri[ords[:,0] == i]))
        Ts = np.append(Ts, sum(Ti[ords[:,0] == i]))
        
    # Compute the reflectance and transmittance at various orders
    efficiencies = [(Rs[i] + Ts[i])/(sum(Rs) + sum(Ts)) for i in range(0,len(orders))]
    return efficiencies


    

def grcwa_transverse_force(nG,cell_geometry,Nx,d,theta,freq):
    '''
    Calculates the reflectance of the infinitely extended unit cell.

        nG: truncation order for spatial discretisation. Affects grating diffraction orders.

        cell_geometry: the geometry of the unit cell, described as either a 1D or 2D array of permittivities.

        Nx: the number of slices used to spatially discretise the unit cell.

        d: width of the unit cell.

        theta: incident angle of light.

    '''
    ## Frequency
    Qabs = np.inf
    freqcmp = freq*(1+1j/2/Qabs)

    ## Unit cell dimension
    L1 = [d,0]

    ## setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta*np.pi/180,phi*np.pi/180,verbose=0)

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
    # input layer information
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 0
    obj.Add_LayerGrid(h,Nx,Ny)           # Layer 1
    obj.Add_LayerUniform(t,E_SiO2)       # Layer 2
    obj.Add_LayerUniform(v_width,E_vacuum)     # Layer 3
    obj.Init_Setup()

    ## Input plane wave and grid information
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
    obj.GridLayer_geteps(cell_geometry.flatten())

    ## Compute reflectance and transmittance by order
    # Ri(Ti) has length obj.nG, to see which order, check obj.G; too see which kx,ky, check obj.kx obj.ky
    Ri,Ti= obj.RT_Solve(byorder=1)
    ords = obj.G # Returns a list of tuples [ord1, ord2] where ord1 is the order in the L1 direction, while ord2 is the order in the L2 direction.
    
    ## Reflectance, transmittance, and efficiencies in orders -1,0,1
    Rs = np.array([])
    Ts = np.array([])
    orders = [0,1,-1]
    for i in orders:
        # Calculate reflection and transmission at angle=theta for various orders i=0,1,-1
        Rs = np.append(Rs, sum(Ri[ords[:,0] == i]))
        Ts = np.append(Ts, sum(Ti[ords[:,0] == i]))
    
    e = (Rs + Ts)/(sum(Ri) + sum(Ti))


    ## Change in efficiency in order -1 as theta changes
    '''
    This can be done in a few ways. Haven't yet decided which one is the best. Also sub-optimal as it requires
    recompiling the grcwa solver.
    '''
    En1_fun = lambda theta: grcwa_efficiencies(nG,orders,cell_geometry,Nx,d,theta,freq)[-1]
    pEn1_pTheta = grad(En1_fun)(0.)
    # numerical_partial = (En1_fun(0.005) - En1_fun(-0.005)) / 0.01
    
    ## Speed of light and wavelength of the incident light
    wavelength = c/freq

    ## Relativistic terms
    beta = v/c
    gamma = 1/np.sqrt(1-beta**2)
    D1 = beta*gamma + gamma - 1

    ## Transverse force
    return -(1/v)*(D1**2)*( 2*(wavelength/d)*pEn1_pTheta*(1/D1 - 1) - (gamma-1)*(2*e[0] + (e[1] + e[-1])*(1 + np.sqrt( 1 - (wavelength/d)**2 ))) )
    

