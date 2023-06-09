#ifndef __GETNODEID_H__
#define __GETNODEID_H__

#include <vector>

int getNodeID(HostDataBlock *host){

	vector <int> Vnodes;

	//char foutNodeID[128];
	//sprintf(foutNodeID, "NodeID.dat");
	FILE*NodeIDout;
	//NodeIDout = fopen(foutNodeID,"w");
	NodeIDout = fopen("./Extra/NodeID.dat","w");

	int Nnodes=host->Nnodes;

	//for loop
	for (int n=0;n<Nnodes;n++){
		real x=host->r[n+0*Nnodes];
		real y=host->r[n+1*Nnodes];
		real z=host->r[n+2*Nnodes];
		//real r=pow(pow(x,2)+pow(y,2),0.5); //polar and not spherical radius!
		//if ( (sqrt(pow(x,2)+pow(y,2)) < 0.5 ) && (z == 2.5) ){
		//if ( (y == 40.0 ) && (z == 5.0) ){

		if ( (x==0.0) && (z == 0.0) ) {
			fprintf(NodeIDout,"%d\t%f\t%f\t%f\n", n, x, y, z);
			fflush(NodeIDout);
			Vnodes.push_back(n);

		}
	}
	//end for

	//printf("\n[ YMG ] size of NodeID is %d\n", Vnodes.size() ); //moved to main.cpp

	fclose(NodeIDout);

	return Vnodes.size();

}

#endif

