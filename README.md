GAFE6_bechmark
==============

GPU based finite element based electrodynamics simulation.

---

##Material Parameters

####All material parameters set within `parameters.h`

* MESHFILE global sets the name of the file the mesh is to be read in from.

---

##Setting Initial Conditions

* Initial velocity and force can be set in `packdata.h` which is where data is copied into data structures which can be sent to the GPU.
* Initial director profile can be set in `setn.h` in the function `setThPh`

---

###Files to be aware of 

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






