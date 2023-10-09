import autograd.numpy as np

def linear_boundary_geometry(Nx,Ny,d,dy,x1,x2,w1,w2):
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

    lower_boundary = boundary_indices[0:2]
    upper_boundary = boundary_indices[2:]

    # Change boundary permittivities
    cell_geometry = cell_geometry.tolist()
    cell_geometry[lower_boundary[0]][0] = eps1
    cell_geometry[lower_boundary[1]][0] = eps1
    cell_geometry[upper_boundary[0]][0] = eps2
    cell_geometry[upper_boundary[1]][0] = eps2
    cell_geometry = np.array(cell_geometry)

    return cell_geometry, eps1, eps2