#ifndef __INITDIRECTOR_H__
#define __INITDIRECTOR_H__

#include "parameters.h"

//function to intialize THETA and PHI before being sent to the simulation
//including the BC's read in from the file
void initN(float *THETA
	, float *PHI
	, int inZ
	, float topBC[inX][inY][2]
	, float botBC[inX][inY][2]){

		
float height;

for(int i=0;i<inX;i++){
	for(int j=0;j<inY;j++){
		for(int k=0;k<inZ;k++){
			
			//intialize by linear interpolation between top and bottom layers
			height = float(k)/float(inZ-1);//should be in range [0 1]
			THETA[(i*inY+ j)*inZ+k] = height*topBC[i][j][0]+(1.0-height)*botBC[i][j][0];
			PHI[(i*inY+ j)*inZ+k] = height*topBC[i][j][1]+(1.0-height)*botBC[i][j][1];
			

		}//k
	}//j
}//i


}//initN



#endif//__INITDIRECTOR_H__