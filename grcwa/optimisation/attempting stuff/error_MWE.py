import numpy as np
from autograd import grad
import nlopt


# DOES NOT WORK AS GRADIENT OUTPUTS TO ZERO IN gradn[:] = grad_fun(x)

# def cost_function(x):
#     cell_geometry = np.ones((10,1), dtype=float)
    
#     if type(x) == npgrad.numpy_boxes.ArrayBox:
#         x = x._value

#     cell_geometry[5:] = x[0]
#     cell_geometry[:5] = x[1]
#
#     z = (x[0])**2 + (x[1])**2
#     return z

def cost_function(x):
    cell_geometry = np.ones((10,1), dtype=float)
    cell_geometry[5:] = x[0] # TypeError: float() argument must be a string or a number, not 'ArrayBox'
    cell_geometry[:5] = x[1] # TypeError: float() argument must be a string or a number, not 'ArrayBox'

    z = (x[0])**2 + (x[1])**2
    print(type(z)) # <class 'autograd.numpy.numpy_boxes.ArrayBox'> whenever grad_fun(x) is passed
    return z

fun = lambda x: cost_function(x)
grad_fun = grad(fun)


def fun_nlopt(x, gradn):
    gradn[:] = grad_fun(x)
    return fun(x)

# set up NLOPT
ndof = 2
init = [5,5]
opt = nlopt.opt(nlopt.LD_MMA, ndof)

opt.set_maxeval(10)
opt.set_max_objective(fun_nlopt)
vars = opt.optimize(init)

print(vars)