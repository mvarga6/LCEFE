#ifndef __SETN_H__
#define __SETN_H__

#include "mainhead.h"
#include "UserDefined.h"
#include "initDirector.h"
#include "printDirector.h"
#include <math.h>

//set theta and phi 
void setThPh(real &theta,real &phi, real rx, real ry, real rz)
{

  theta = PI/2.0 - 0.1f;
  phi = 0;

}//setThPh


//set the theta and phi corresponding to the average director inside each tetrahedra
//simple now but will connect this to the mathematica GUI
//****must execute get_tet_pos first*****
void set_n(TetArray &Tets, SimulationParameters *params)
{
	int Ntets = Tets.size;
	real rz, theta=0.0, phi=0.0; //, rx, ry
	const real min = Tets.min(2), max = Tets.max(2);
	const real range = max - min;
	real u;
	for(int i = 0; i < Ntets; i++){

		//get position of tetrahedra
//		rx = i_Tet.get_pos(i,0);
//		ry = i_Tet.get_pos(i,1);
		rz = Tets.get_pos(i,2);

		u = (rz - min) / range;
		
		// notes
		// phi is in xy plane
		// theta is from the z axis
		
		//if (params->Initalize.PlanarSideUp) theta = u*(PI/2.0);
		//else theta = (1.0-u)*(PI/2.0); // homeotropic top
		
		theta = PI/2.0;
		phi = 0;

		//turn positions into director
    		//setThPh(theta,phi,rx,ry,rz);

		//assign theta and phi to tetrahedra
		Tets.set_theta(i,theta);
		Tets.set_phi(i,phi);
	}//i


}//set angle


#endif//__SETN_H__
