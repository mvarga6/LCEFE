#ifndef __SETN_H__
#define __SETN_H__

#include "mainhead.h"
#include "UserDefined.h"
#include "initDirector.h"
#include "printDirector.h"
#include <math.h>

//set theta and phi 
void setThPh(float &theta,float &phi, 
			float rx, float ry, float rz, 
			float rx_min, float rx_range,
			float ry_min, float ry_range,
			float rz_min, float rz_range){
	
	float x = (rx - rx_min) / rx_range;
	float y = (ry - ry_min) / ry_range;
	float z = (rz - rz_min) / rz_range;
	
	theta = PI * z - PI / 2.0f;
	phi = PI / 2.0f;
}//setThPh


//set the theta and phi corresponding to the average director inside each tetrahedra
//simple now but will connect this to the mathematica GUI
//****must execute get_tet_pos first*****
void set_n(TetArray *i_Tet,int Ntets){
	
	float rx, ry, rz, theta = 0.0, phi = 0.0;
	float rx_min = i_Tet->min(AXIS::X);
	float ry_min = i_Tet->min(AXIS::Y);
	float rz_min = i_Tet->min(AXIS::Z);
	float rx_range = i_Tet->max(AXIS::X) - rx_min;
	float ry_range = i_Tet->max(AXIS::Y) - ry_min;
	float rz_range = i_Tet->max(AXIS::Z) - rz_min;
	for(int i = 0; i < Ntets; i++){

		//get position of tetrahedra
		rx = i_Tet->get_pos(i, 0);
		ry = i_Tet->get_pos(i, 1);
		rz = i_Tet->get_pos(i, 2);
		
		//turn positions into director
		setThPh(theta, phi, rx, ry, rz, rx_min, rx_range, ry_min, ry_range, rz_min, rz_range);

		//assign theta and phi to tetrahedra
		i_Tet->set_theta(i, theta);
		i_Tet->set_phi(i, phi);
	}//i
}//set angle


#endif//__SETN_H__
