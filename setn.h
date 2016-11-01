#ifndef __SETN_H__
#define __SETN_H__

#include "mainhead.h"
#include "UserDefined.h"
#include "initDirector.h"
#include "printDirector.h"
#include <math.h>

//set theta and phi 
void setThPh(float &theta,float &phi, float rx, float ry, float rz){

  theta = PI/2.0;
  phi = 0;

}//setThPh


//set the theta and phi corresponding to the average director inside each tetrahedra
//simple now but will connect this to the mathematica GUI
//****must execute get_tet_pos first*****
void set_n(TetArray &i_Tet,int Ntets){
	float rx,ry,rz,theta=0.0,phi=0.0;
	
	for(int i=0;i<Ntets;i++){

		//get position of tetrahedra
		rx = i_Tet.get_pos(i,0);
		ry = i_Tet.get_pos(i,1);
		rz = i_Tet.get_pos(i,2);
		
		//turn positions into director
    		setThPh(theta,phi,rx,ry,rz);

		//assign theta and phi to tetrahedra
		i_Tet.set_theta(i,theta);
		i_Tet.set_phi(i,phi);
	}//i


}//set angle


#endif//__SETN_H__
