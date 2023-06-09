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
void getThPhi(real rx				// x position [cm]
			, real ry				// y position [cm]
			, real rz				// z position [cm]
			, real &theta			// theta [rad] (reference return)
			, real &phi			// phi [rad] (reference return)
			, real * THETA 		// top boundary conditions
			, real * PHI			// bottom boundary conditions
			, int inZ
			, real xMAX
			, real xMIN
			, real yMAX
			, real yMIN
			, real zMAX
			, real zMIN
			 ){
//-----------------------------------------------------------//
//                 WRITE YOUR CODE HERE                      //

//convert rx and ry into integrers and rz into top or bottom

	int i,j,k;

	i = int(real(inX)*(rx+xMIN)/(xMAX-xMIN));
	j = int(real(inY)*(ry+yMIN)/(yMAX-yMIN));
	k = int(real(inZ)*(rz+zMIN)/(zMAX-zMIN));

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
__device__ void userForce(real *r                // node pos
			, real (&F)[12]          // node-force tensor
			, real xmax
						 ){
//-----------------------------------------------------------//
//                 WRITE YOUR CODE HERE                      //
/*	for(int n = 0; n < 4; n++){*/
/*		for(int i = 0; i < 3; i++){*/
/*			if(r[i+n*3] > xmax - 0.5f){*/
/*				F[i+n*3] -= cxxxx*0.00001f;*/
/*			}*/
/*		}*/
/*	}*/

//-----------------------------------------------------------//
}//userForce







#endif// __USERDEFINED_H__
