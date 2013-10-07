#ifndef __USERDEFINED_H__
#define __USERDEFINED_H__

#include <math.h>
#include "mainhead.h"
#include "parameters.h"


//===========================================================//
//                  Set Director Profile                     //
//                                                           //
//	Input:  rx, ry and rz  [cm]                              //
//	       (centriod of tetrahedra)                          //
//                                                           //
//  Output: theta and phi [rad]                              //
//         (returned as refference)                          //
//                                                           //
//===========================================================//
void getThPhi(float rx				// x position [cm] 
			, float ry				// y position [cm]
			, float rz				// z position [cm]
			, float &theta			// theta [rad] (reference return)
			, float &phi			// phi [rad] (reference return)
			, float * THETA 		// top boundary conditions
			, float * PHI			// bottom boundary conditions
			, int inZ
			, float xMAX
			, float xMIN
			, float yMAX
			, float yMIN
			, float zMAX
			, float zMIN
			 ){
//-----------------------------------------------------------//
//                 WRITE YOUR CODE HERE                      //

//convert rx and ry into integrers and rz into top or bottom

	int i,j,k;

	i = int(float(inX)*(rx+xMIN)/(xMAX-xMIN));
	j = int(float(inY)*(ry+yMIN)/(yMAX-yMIN));
	k = int(float(inZ)*(rz+zMIN)/(zMAX-zMIN));

	theta = THETA[(i*inY+ j)*inZ+k];
	phi = PHI[(i*inY+ j)*inZ+k];

//-----------------------------------------------------------//
}//getThPhi



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////



//===========================================================//
//                   User Defined Force                      //
//                                                           //
//	Input:  r[12]: xyz positions of nodes in tetrahedra [cm] //
//          Q[9]: Q-tensor                                   //
//          F[12]: Current forces to add to and return       //
//          TetNodeRank[4]:  rank of nodes in tetrahedra     //
//                                                           //
//-----------------------------------------------------------//
//                                                           //
//   Coordinates of nth node:  x = r[n*3]                    //
//       (same for F)          y = r[n*3+1]                  //
//                             z = r[n*3+2]                  //
//                                                           //
//   Coordinates of Q-tensor:  Q(i,j) = Q[i*3+j]             //
//                                                           //
//-----------------------------------------------------------//
//                                                           //
//   Output: F[12] refference return                         //
//                                                           //
//===========================================================//
__device__ void userForce(float *r                // node positions
						, float *Q                // Q tensor 
						, float (&F)[12]          // node-force tensor
						 ){
//-----------------------------------------------------------//
//                 WRITE YOUR CODE HERE                      //




//-----------------------------------------------------------//
}//userForce







#endif// __USERDEFINED_H__