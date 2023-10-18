'''
This file optimises for transverse force by adjusting the width AND position of each grating element. Here we do a beam search.
'''
import core
import autograd.numpy as np
import numpy
from autograd import grad
import matplotlib.pyplot as plt

## Discretisation values
nG = 30
Nx = 1801

## Sail speed
beta = 0.2

## Ilic cell parameters
d = 1.8

# Permittivities
E_Si = 3.5**2
E_SiO2 = 1.45**2
E_vacuum = 1.

## Light incidence
c = 1.
wavelength = 1.5
freq = c/wavelength
theta = 0.

'''
The beam parameter space is the width and position of each grating element:

    x1: the central position of element 1 in the unit cell

    x2: the central position of element 2 in the unit cell

    w1: the width of element 1 in the unit cell

    w2: the width of element 2 in the unit cell
'''

param_lims = [[0, 0.5*d],   # x1 limits
              [0, d],   # w1 limits
              [0.5*d, 0],   # x2 limits
              [0, d]    # w2 limits
              ]

num_beams = 10

x1s = numpy.linspace(param_lims[0][0], param_lims[0][1], num_beams)
w1s = numpy.linspace(param_lims[1][0], param_lims[1][1], num_beams)
x2s = numpy.linspace(param_lims[2][0], param_lims[2][1], num_beams)
w2s = numpy.linspace(param_lims[3][0], param_lims[3][1], num_beams)