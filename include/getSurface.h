#ifndef __GETSURFACE_H__
#define __GETSURFACE_H__

#include <vector>


std::vector<int> getSurface(HostDataBlock *host){
	int Nnodes=host->Nnodes;
	int Ntets=host->Ntets;


	vector <int> SurfaceList;
	vector <int> SurfaceNodeList;

	//char foutNodeID[128];
	//sprintf(foutNodeID, "NodeID.dat");
	FILE*SurfIDout;
	//NodeIDout = fopen(foutNodeID,"w");
	SurfIDout = fopen("./Extra/Surface/SurfID.dat","w");

	//STEP 1: Define your surface and get surface node IDs
	for (int n=0;n<Nnodes;n++){
		real x=host->r[n+0*Nnodes];
		real y=host->r[n+1*Nnodes];
		real z=host->r[n+2*Nnodes];
		//real r=pow(pow(x,2)+pow(y,2),0.5); //polar and not spherical radius!
		//if ( (sqrt(pow(x,2)+pow(y,2)) < 0.5 ) && (z == 2.5) ){
		//if ( (y == 40.0 ) && (z == 5.0) ){

		if ( (z == 2.0) ) {
			fprintf(SurfIDout,"%d\t%f\t%f\t%f\n", n, x, y, z);
			fflush(SurfIDout);
			SurfaceNodeList.push_back(n);
			SurfaceList.push_back(n);

		}
	}
	fclose(SurfIDout);
	//End STEP 1

	//////////////////////////

	//STEP 2:which tets share those surface nodes (count thru Ntets)
	for (int t=0; t < Ntets; t++)
	{
		/*
		printf("\n[ Tet Test ] Tet %d has %d\t%d\t%d\t%d",
		t, host->TetToNode[t+0*Ntets], host->TetToNode[t+1*Ntets], host->TetToNode[t+2*Ntets], host->TetToNode[t+3*Ntets] );
		*/
	}

	vector <int> countSurfNodes (Ntets, 0);
	//std::fill(countSurfNodes.begin(), countSurfNodes.end(), 0);

	int Ntriangles = 0;
	for (int t = 0; t < Ntets; t++)
	{
		for (int sweep = 0; sweep < 4; sweep++)
		{
			int NofT = host->TetToNode[t+sweep*Ntets];
			for(int s = 0; s < SurfaceNodeList.size(); s++)
			{
				if (SurfaceNodeList[s] == NofT) {countSurfNodes[t]++;}
			}
		}

		if (countSurfNodes[t] == 3) {Ntriangles++;}
		//printf("\n[ Count ] Tet %d counts %d", t, countSurfNodes[t]); //debug
	}
	printf("\n[ YMG ] There are %d triangles", Ntriangles);
	//End STEP 2
	//////////////////////////

	//STEP 3 create 3-peice wise array of surface nodes
	FILE*TriToNode;
	TriToNode = fopen("./Extra/Surface/TriToNode.dat","w");

	vector <int> surfTrinagles ; //negative so we know if sth not assigned
	for (int t = 0; t < Ntets; t++)
	{
		if (countSurfNodes[t] == 3) {
			//printf ("\nAlright boss...tet %d has 3 surf node", t);
			fprintf(TriToNode,"%d",t);

			for (int sweep = 0; sweep < 4; sweep++)
			{
				int NofT = host->TetToNode[t+sweep*Ntets];
				for(int s = 0; s < SurfaceNodeList.size(); s++)
					{
						if (SurfaceNodeList[s] == NofT) {
							surfTrinagles.push_back(NofT);
							fprintf(TriToNode,"\t%d",NofT);
						}
					}
			}


			fprintf(TriToNode,"\n");
		}

	}
	fclose(TriToNode);
	//End STEP 3
	//////////////////////////


	//Create the SurfList
	for (int t = 0; t < Ntets; t++)
	{
		if (countSurfNodes[t] == 3) {
			SurfaceList.push_back(t);


			for (int sweep = 0; sweep < 4; sweep++)
			{
				int NofT = host->TetToNode[t+sweep*Ntets];
				for(int s = 0; s < SurfaceNodeList.size(); s++)
					{
						if (SurfaceNodeList[s] == NofT) {
							SurfaceList.push_back(NofT);

						}
					}
			}



		}

	}
	//creatd SurfaceList
	//////////////////////////

	return SurfaceList;
}

#endif

