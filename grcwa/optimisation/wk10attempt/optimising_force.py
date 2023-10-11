import core
import autograd.numpy as np
from autograd import grad

## Discretisation values
nG = 30
Nx = 181

## Sail speed
beta = 0.2

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
theta = 0.



## Objective function to minimise
min_objective = lambda vars: core.grcwa_transverse_force(nG, core.linear_permittivity_geometry(Nx,d,x1,x2,w1,w2,vars), Nx, d, theta, freq, beta)
grad_fun = grad(min_objective)

## Optimisation
# Initialisation parameters
vars = [5.,5.]                                 # [eps1, eps2]
step_size = 1/np.sum(np.array(grad_fun(vars))) # define step size so that first step is of size 1
obj_vals = []
var_vals = []
# Optimisation loop
step = 0
optimising = True
while optimising:
    ## Print initial values
    if step == 0:
        obj = min_objective(vars)
        print("{:<8} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8}".format("Step", "Objective", 'eps1', 'eps2', 'w1', 'w2'))
        print("----------------------------------------------------------------")
        print("{:<8} | {:<10.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format(step,obj,vars[0],vars[1],w1,w2))

        # Add to history of parameters and objective values
        var_vals.append(vars)
        obj_vals.append(obj)

    ## Perform Optimisation
    else:
        if step % 8 == 0:
            print("----------------------------------------------------------------")
            print("{:<8} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8}".format("Step", "Objective", 'eps1', 'eps2', 'w1', 'w2'))
            print("----------------------------------------------------------------")

        # Calculate current gradient
        grad = grad_fun(vars)

        # Step in direction of local minimum
        vars = vars - step_size*np.array(grad)
        # Ensure that the permittivities are from E_vacuum to E_Si
        vars, w1, w2 = core.valid_eps(vars, d, Nx, w1, w2)
        
        # Calculate cost function after stepping
        obj = min_objective(vars)
        
        # Condition 1: minimum has been overstepped, leading objective to increase
        if obj > obj_vals[-1]:
            step_size = 0.9*step_size # decrease the step size
            continue

        # Condition 2: recent values are identical
        elif max(obj_vals[-5:]) - min(obj_vals[-5:]) < 1e-3:
            step_size = 1.5*step_size # increase step size for next time
            

        

        # Add to history of parameters and objective values
        var_vals.append(vars)
        obj_vals.append(obj)

        ## Print current state to terminal
        print("{:<8} | {:<10.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format(step,obj,vars[0],vars[1],w1,w2))
            
                

    step += 1 # move to next step in optimisation




'''
Objectives:
    1) Ensure that cell_geometry is dynamically changing        DONE
    
'''