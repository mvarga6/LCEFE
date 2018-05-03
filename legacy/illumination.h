#ifndef __ILLUMINATION_H__
#define __ILLUMINATION_H__
#include "mainhead.h"
#include "parameters.h"
#include <math.h>


//.. for each surface triangle find the point of illumination
__global__ void point_of_illumination_kernel(
	real (&illum_vector)[3] // incident wave vector
	,int &cell_illum_from	 // cell_id where light emanates from
	){
	
	// 1) Grab surface triangle node positions
	
	// 2) Trace ray from triangle pos to wave front plane
	//	  in -k_hat direction (k: normal of incident light).
	
	// 3) Find cell (on wave front plane) which light must emanate 
	//    from and mark id in list. Each tetra is (possibly) hit by 
	//    light from discrete parts of wave front, each with unique 
	//    ids.  Each tet gets an id telling where the light came from.
}

#endif
