import core
import autograd.numpy as np
from autograd import grad

## Discretisation values
nG = 30
Nx = 1801

## Sail speed
beta = 0.2

## Ilic cell parameters
d = 1.8
x1 = 0.25*d
x2 = 0.85*d
w1 = 0.35*d
w2 = 0.15*d
eps1 = 5.
eps2 = 5.
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
vars = [eps1,eps2]
step_size = 1/np.sqrt(np.sum(np.array(grad_fun(vars))**2)) # define step size so that first step is of size 1
obj_vals = []
var_vals = []
w_vals = []
# Optimisation loop
print(r"Ready to optimise with BETA = {}, nG = {}, Nx = {}".format(beta,nG,Nx))
step = 0
overstep_count = 0
optimising = True
while optimising:
    ## Print initial values
    if step == 0:
        obj = min_objective(vars)
        print("{:<8} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8}".format("Step", "Objective", 'eps1', 'eps2', 'w1', 'w2'))
        print("----------------------------------------------------------------")
        print("{:<8} | {:<10.5f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format(step,obj,vars[0],vars[1],w1,w2))

        # Add to history of parameters and objective values
        var_vals.append(vars)
        obj_vals.append(obj)
        w_vals.append([w1,w2])

    ## Perform Optimisation
    else:
        if step % 8 == 0:
            print("----------------------------------------------------------------")
            print("{:<8} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8}".format("Step", "Objective", 'eps1', 'eps2', 'w1', 'w2'))
            print("----------------------------------------------------------------")

        # Calculate current gradient
        grad = grad_fun(vars)

        # Step in direction of local minimum
        vars = vars + step_size*np.array(grad)
        # Ensure that the permittivities are from E_vacuum to E_Si
        vars, w1, w2 = core.valid_eps(vars, d, Nx, w1, w2)
        # Calculate cost function after stepping
        obj = min_objective(vars)
        
        # Condition 1: minimum has been overstepped, leading objective to increase
        if obj - obj_vals[-1] > 1e-5:
            overstep_count += 1
            if overstep_count > 20:
                print("Optimum reached after {} steps.".format(step))
                break
            else:
                step_size = 0.9*step_size # decrease the step size
                print("{:<8} | {:<10.5f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format('OVERSTEP',obj,vars[0],vars[1],w1,w2))
                
                vars = var_vals[-1] # move vars back to their original place
                w1 = w_vals[-1][0]
                w2 = w_vals[-1][1]
                continue
        else:
            overstep_count = 0

        # Condition 2: recent values are identical
        if step > 5 and max(obj_vals[-2:]) - min(obj_vals[-2:]) < 1e-3:
            step_size = 2*step_size # increase step size for next time
            

        

        # Add to history of parameters and objective values
        var_vals.append(vars)
        obj_vals.append(obj)
        w_vals.append([w1,w2])

        ## Print current state to terminal
        print("{:<8} | {:<10.5f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format(step,obj,vars[0],vars[1],w1,w2))
            
                

    step += 1 # move to next step in optimisation

basic_geometry = core.basic_cell_geometry(Nx,d,x1,x2,w1,w2)
basic_cost = core.grcwa_transverse_force(nG, basic_geometry, Nx, d, theta, freq, beta)
print("Cost without boundary permittivities:", basic_cost)

'''
Objectives:
    1) (11/10/23) Ensure that cell_geometry is dynamically changing                                    DONE on 11/10/23

    2) (11/10/23) Fix eps going out of bounds with beta = 0.02                                         DONE on 11/10/23

    3) (11/10/23) Fix overstepping display. The vars[0] and vars[1] variables should be changing       DONE on 11/10/23

    4) (11/10/23) Fix gradient direction for beta = 0.02 vs beta = 0.2. I keep having to change 
    vars = vars + ... into vars = vars - ... to approach the minimum. It should be
    vars = vars - ..., but beta = 0.02 is completely reversed (incorrect).

    5) Implement a beam search
'''