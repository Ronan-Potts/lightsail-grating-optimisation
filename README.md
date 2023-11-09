# Optimising Gratings for LightSails

### What are we optimising?
Want damping for the transverse and rotational motion of the lightsail. 
* We can get this with diffraction gratings, but we need to be able to calculate the torques and forces on the sail for an arbitrary diffraction grating.
* Need to use numerical methods to calculate torques and forces. Various softwares allow us to do this.

### Computational Analysis

We are looking at three main softwares: GRCWA, MEEP, and Tidy3D.

* **GRCWA (autoGradable Rigorous Coupled-Wave Analysis):** uses rigorous coupled-wave analysis to solve Maxwell's equations in an arbitrary grating which extends infinitely. Solves Maxwell's equations on the basis that a plane wave is incident on the diffraction grating. Also has autoGradable capabilities, allowing for gradient optimisation with the AVM (adjoint variable method).

* **Meep:** uses FDTD (finite-difference time domain), discretising time to solve Maxwell's equations. This is less "intelligent" than GRCWA, but allows for more freedom in implementation, such as the use of a laser which is not a plane wave, or a diffraction grating of finite size.

* **Tidy3D:** also uses FDTD, like Meep. Tidy3D extends Meep by incorporating the adjoint variable method to allow for optimisation of gratings, given a cost function.
