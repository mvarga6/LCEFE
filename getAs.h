#ifndef __GETAS_H__
#define __GETAS_H__

#include "invert4x4.h"

//calculate current A matrix and then invert it
//inverted A and original A stored in Tet object
void init_As(NodeArray &i_Node,TetArray &i_Tet,int Ntets){
	float tempAin[16],tempAout[16];
	int ip;

	for(int t=0;t<Ntets;t++){
		for(int i = 0;i<4;i++){

				//get particle number of that tetrahedra
				ip = i_Tet.get_nab(t,i);

				//find A matrix
				
				tempAin[4*i] = 1.0;
				tempAin[4*i+1] = i_Node.get_pos(ip,0);
				tempAin[4*i+2] = i_Node.get_pos(ip,1);
				tempAin[4*i+3] = i_Node.get_pos(ip,2);


				//printf("A1 = %f A2 = %f A3 = %f A4 = %f\n",tempAin[4*i],tempAin[4*i+1],tempAin[4*i+2],tempAin[4*i+3]);

				//Store A matrix in object
				i_Tet.set_A(t,i,0,tempAin[4*i]);
				i_Tet.set_A(t,i,1,tempAin[4*i+1]);
				i_Tet.set_A(t,i,2,tempAin[4*i+2]);
				i_Tet.set_A(t,i,3,tempAin[4*i+3]);

		}//i

				//Cacluate inverse of A matrix
				if(!gluInvertMatrix(tempAin,tempAout)){
					printf("Matrix inversion failed; Determinant = 0\n");
				}else{
					//printf("A1 = %f A2 = %f A3 = %f A4 = %f\n",tempAin[4*i],tempAin[4*i+1],tempAin[4*i+2],tempAin[4*i+3]);
					for(int i = 0;i<4;i++){
					i_Tet.set_invA(t,i,0,tempAout[4*i]);
					i_Tet.set_invA(t,i,1,tempAout[4*i+1]);
					i_Tet.set_invA(t,i,2,tempAout[4*i+2]);
					i_Tet.set_invA(t,i,3,tempAout[4*i+3]);
					}
				}


				
		
	}//t

	printf("A's calculated, inverted and stored.\n");
}

#endif //__GETAS_H__
