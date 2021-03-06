#ifndef __GETAS_H__
#define __GETAS_H__

#include "invert4x4.h"

/// Calculate current A matrix and then invert it
/// inverted A and original A stored in Tet object
static void init_As(NodeArray &Nodes, TetArray &Tets)
{
	int Ntets = Tets.size;
	//int Nnodes = Nodes.size;
	real tempAin[16],tempAout[16];
	int ip;

	for(int t = 0; t < Ntets; t++)
	{
		for(int i = 0; i < 4; i++)
		{
				//get particle number of that tetrahedra
				ip = Tets.get_nab(t,i);

				//find A matrix
				tempAin[4*i] = 1.0;
				tempAin[4*i+1] = Nodes.get_pos(ip,0);
				tempAin[4*i+2] = Nodes.get_pos(ip,1);
				tempAin[4*i+3] = Nodes.get_pos(ip,2);


				//printf("A1 = %f A2 = %f A3 = %f A4 = %f\n",tempAin[4*i],tempAin[4*i+1],tempAin[4*i+2],tempAin[4*i+3]);

				//Store A matrix in object
				Tets.set_A(t, i, 0, tempAin[4*i]);
				Tets.set_A(t, i, 1, tempAin[4*i+1]);
				Tets.set_A(t, i, 2, tempAin[4*i+2]);
				Tets.set_A(t, i, 3, tempAin[4*i+3]);
		}//i

				//Cacluate inverse of A matrix
		if (!gluInvertMatrix(tempAin, tempAout))
		{
			printf("\nMatrix inversion failed; Determinant = 0");
			printf("\nTet: %d Nodes: [%d %d %d %d]", t, 
				Tets.get_nab(t, 0), 
				Tets.get_nab(t, 1),
				Tets.get_nab(t, 2),
				Tets.get_nab(t, 3));
		}
		else
		{
			//printf("A1 = %f A2 = %f A3 = %f A4 = %f\n",tempAin[4*i],tempAin[4*i+1],tempAin[4*i+2],tempAin[4*i+3]);
			for(int i = 0; i < 4; i++)
			{
				Tets.set_invA(t, i, 0, tempAout[4*i]);
				Tets.set_invA(t, i, 1, tempAout[4*i+1]);
				Tets.set_invA(t, i, 2, tempAout[4*i+2]);
				Tets.set_invA(t, i, 3, tempAout[4*i+3]);
			}
		}
	}//t

	printf("\nA's calculated, inverted and stored.");
}

#endif //__GETAS_H__
