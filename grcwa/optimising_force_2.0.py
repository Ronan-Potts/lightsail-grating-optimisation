'''
This file optimises for transverse force by adjusting the width AND position of each grating element.
'''
import core
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

save_figs = True

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


## Visualisation
plt.ion()
fig1, ax1 = plt.subplots()


## Objective function to minimise
min_objective = lambda vars: core.grcwa_transverse_force(nG, core.independent_boundary_geometry(Nx,d,x1,x2,w1,w2,vars[0],vars[1],vars[2],vars[3]), Nx, d, theta, freq, beta)
grad_fun = grad(min_objective)

## Optimisation
# Initialisation parameters
vars = [eps1,eps2,eps1,eps2] # top of e1, top of e2, bottom of e1, bottom of e2
step_size = 1/np.sqrt(np.sum(np.array(grad_fun(vars))**2)) # define step size so that first step is of size 1
obj_vals = []
all_objs = []
var_vals = []
w_vals = []
x_vals = []

# Optimisation loop
print(r"Ready to optimise with BETA = {}, nG = {}, Nx = {}".format(beta,nG,Nx))
step = 0
overstep_count = 0
optimising = True
while optimising:
    ## Print initial values
    if step == 0:
        obj = min_objective(vars)
        print("{:<13} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format("Step", "Objective", 'eps1', 'eps2', 'x1', 'x2', 'w1', 'w2'))
        print("--------------------------------------------------------------------------------------")
        print("{:<13} | {:<10.5f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format(step,obj,vars[0],vars[1],x1,x2,w1,w2))

        # Add to history of parameters and objective values
        var_vals.append(vars)
        all_objs.append(obj)
        obj_vals.append(obj)
        x_vals.append([x1,x2])
        w_vals.append([w1,w2])

        
        cell_geometry = core.independent_boundary_geometry(Nx,d,x1,x2,w1,w2,vars[0],vars[1],vars[2],vars[3])

        ## Plot data
        anim1 = ax1.imshow(cell_geometry, interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
        cbar = plt.colorbar(anim1)
        cbar.set_label("Permittivity")
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title(r"Initialisation, Force Component = {} $I A' v_y / c$ with $\beta = {}$".format(round(obj,3),beta))
        if save_figs:
            plt.savefig('grcwa/figs/INITIAL_optimising_force_2.0.png')

    ## Perform Optimisation
    else:
        if step % 8 == 0 and overstep_count == 0:
            print("-----------------------------------------------------------------------------------------")
            print("{:<13} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format("Step", "Objective", 'eps1', 'eps2', 'x1', 'x2', 'w1', 'w2'))
            print("-----------------------------------------------------------------------------------------")

        # Calculate current gradient
        grad = grad_fun(vars)

        # Step in direction of local minimum
        vars = vars - step_size*np.array(grad)

        # Ensure that the permittivities are in range [E_vacuum, E_Si]
        while core.valid(vars) == False:
            vars, x1, x2, w1, w2 = core.valid_eps(vars, d, Nx, x1, x2, w1, w2)

        # Calculate cost function after stepping
        obj = min_objective(vars)
        

        # Condition 2: recent values are identical
        if step > 5 and max(obj_vals[-2:]) - min(obj_vals[-2:]) < 1e-3:
            step_size = 2*step_size # increase step size for next time
            
        # Condition 1: minimum has been overstepped, leading objective to increase
        if obj - obj_vals[-1] > 1e-5:
            overstep_count += 1
            all_objs.append(obj)
            if overstep_count > 20:
                print("\nOptimum reached after {} steps.".format(step))
                break
            else:
                step_size = 0.9*step_size # decrease the step size
                print("{:<13} | {:<10.5f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format('OVERSTEP #{}'.format(overstep_count),obj,vars[0],vars[1],x1,x2,w1,w2), end='\r')
                continue
        else:
            overstep_count = 0
            

        # Add to history of parameters and objective values
        var_vals.append(vars)
        all_objs.append(obj)
        obj_vals.append(obj)
        x_vals.append([x1,x2])
        w_vals.append([w1,w2])

        ## Print current state to terminal
        print("{:<13} | {:<10.5f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f} | {:<8.3f}".format(step,obj,vars[0],vars[1],x1,x2,w1,w2))
            
        ## Update figure
        cell_geometry = core.independent_boundary_geometry(Nx,d,x1,x2,w1,w2,vars[0],vars[1],vars[2],vars[3])
        anim1.set_data(cell_geometry)
        plt.title(r"Step {}, Force Component = {} $I A' v_y / c$ with $\beta = {}$".format(step, round(obj,3),beta))
        fig1.canvas.flush_events()

    step += 1 # move to next step in optimisation

vars = var_vals[-1]
x1,x2 = x_vals[-1]
w1,w2 = w_vals[-1]
basic_geometry = core.basic_cell_geometry(Nx,d,x1,x2,w1,w2)
basic_cost = core.grcwa_transverse_force(nG, basic_geometry, Nx, d, theta, freq, beta)
print("Cost without boundary permittivities:", basic_cost)

plt.imshow(basic_geometry, interpolation='nearest', vmin=0, vmax=E_Si, aspect='auto')
plt.xlabel("y")
plt.ylabel("x")
plt.title(r"Final, Step {}, Force Component = {} $I A' v_y / c$ with $\beta = {}$".format(step, round(basic_cost,3),beta))
if save_figs:
    plt.savefig('grcwa/figs/FINAL_optimising_force_2.0.png')



plt.close()

## Mapping out objective function
all_objs.sort()
steps = range(0,len(all_objs))

plt.plot(steps, all_objs)
plt.savefig('grcwa/figs/optimising_force_2.0.png')

'''
Objectives:
    1) Incorporate imshow                     DONE
'''