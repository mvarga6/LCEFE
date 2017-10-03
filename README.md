Liquid Crystal Elastomer Finite Element 
=======================================
GPU accelerated finite element based electrodynamics simulation.

Written for research at Liquid Crystal 
Institute @ Kent State University, Kent OH

#### Authers: 

** Andrew Konya
** Michael Varga

---

## Building

* Build using `make`
* This compiles all object files and links to executable in builds/. 
* The name of the build is set by $(BUILDNAME) found on the first like of `makefile` 

## Runtime Parameters

* The executable takes cmdline arguments to set parameters in the simulation. It also accepts a parameters file to read the values from.
* An example parameters file `params.json` is provided. It reads standard json or the simplified json found in the example.
* A list of available cmdline options and parameters are shown below...

#### Material Parameters

`{name} : {type} : {description}` 

`density : float : Sets the material density of the elastomer` 

`alpha : float : Sets the compling stength of nematic order to stress` 

`cxxxx : float : Primary elastic constant` 

`cxxyy : float : Secondary elastic constant` 

`cxyxy : float : Elastic constant` 

#### Simulation Parameters

`{name} : {type} : {description}` 

`nsteps : int : The number of timesteps to run the simulation` 

`dt : float : The length of time steps` 

`framerate : int : how often to print output files` 

### Files to be aware of 

`classstruct.h` defines both `TetArra`y and `NodeArray` classes which are used to optimize the mesh on the CPU side.

`getmesh.h` parses the mesh (`get_mesh`), as well as optimizes the mesh ordering (`gorder_tet`).

`packdata.h` copies all mesh and initial condition data into arrays which can be coppied to the GPU

`rundynamics.h` handles the entire dynamics loop, including the invocation of the GPU side kernels.

`gpuForce.h` is the GPU side Kernel to handle force calculations, and uses:
* `forcecalc.h` to do the actual force calculation
* `read_dev_data.h` to get global GPU data needed for the force calculation
* `getQ.h` is used to get the LC director for a given element
* `sendForce.h` sends the forces calculated within the kernel to global memory

`updateKernel.h` is the GPU kernel to handle updating node positions and uses:
* `update_r.h` to do the actual update equation calculations
* `sumForce.h` is used to read and sum forces stored in global memory






