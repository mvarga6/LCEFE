#ifndef __GETMESH_H__
#define __GETMESH_H__

#include "genrand.h"
#include "mainhead.h"
#include <math.h>


//====================================================
// calculate the volume of a tetrahedra
//====================================================
float tetVolume(float x1, float y1, float z1
			   ,float x2, float y2, float z2
			   ,float x3, float y3, float z3
			   ,float x4, float y4, float z4){


float a11=x1-x2;
float a12=y1-y2;
float a13=z1-z2;

float a21=x2-x3;
float a22=y2-y3;
float a23=z2-z3;

float a31=x3-x4;
float a32=y3-y4;
float a33=z3-z4;
float vol0=a11*a22*a33+a12*a23*a31+a13*a21*a32;
      vol0=vol0-a13*a22*a31-a11*a23*a32-a12*a21*a33;
      vol0=vol0/6.0;

return abs(vol0);

	

}//tet volume


//read the mesh from mesh.dat
void get_mesh_dim(int &Ntets, int &Nnodes, const std::string &meshFile = DEFAULTMESHFILE){
	int d1A,d2,d3,d4,d5,d1B;
	float f0,f1,f2;
	int Ttot;
	FILE* meshin;
    meshin = fopen(meshFile.c_str(),"r");

	//read total number of nodes and tetrahedras
	fscanf(meshin,"%d %d\n",&Nnodes,&Ttot);



	for(int i=0;i<Nnodes;i++){
		fscanf(meshin,"%d %e %e %e",&d2,&f0,&f1,&f2);
	}
	int count=0;
	int go=1;
	int next=0;
	d1B=0;
	for(int i=0;i<Ttot;i++){

		if (next==0){
		fscanf(meshin,"%d %d %d %d",&d1A,&d2,&d3,&d4);

		if(d1B+1==d1A){
			d1B=d1A;
			count++;
		}else{
			next=1;
			d1B=d2;

			fscanf(meshin,"%d %d",&d3,&d4);
			count++;
		}
		
		}
		if(next==1){
			fscanf(meshin,"%d %d %d %d %d",&d1A,&d2,&d3,&d4,&d5);
			if(d1B+1==d1A&&go==1){
				d1B=d1A;
				count++;
			}else{
				go=0;
			}
		}

	}


	Ntets = Ttot-count;
	fclose(meshin);
	printf("Mesh loaded \n Nodes=%d \n Tetrahedra = %d\n",Nnodes,Ntets);
}

void get_mesh(NodeArray &i_Node,TetArray &i_Tet,int Ntets, int Nnodes, const std::string &meshFile = DEFAULTMESHFILE){
	int dummy,Ttot;
	float rx,ry,rz;
	int n0,n1,n2,n3;

	//open file to read mesh data from
	FILE* meshin;
    meshin = fopen(meshFile.c_str(),"r");

	//read total number of nodes and tetrahedras
	fscanf(meshin,"%d %d\n",&dummy,&Ttot);



	//scan in node positions
	for(int i=0;i<Nnodes;i++){
		fscanf(meshin,"%d %e %e %e\n",&dummy,&rx,&ry,&rz);
		i_Node.set_pos(i,0,rx*meshScale);
		i_Node.set_pos(i,1,ry*meshScale);
		i_Node.set_pos(i,2,rz*meshScale);
	}

	//scan over the edge lines and faces
	//right now we only want the informatino
	//about each tetrahedra
	char dummystring[100];
	for(int i=0;i<(Ttot-Ntets);i++){
		fgets(dummystring,100,meshin);
	}

	//scan in Tetrahedra 
	for(int i=0;i<Ntets;i++){
		fscanf(meshin,"%d %d %d %d\n",&n0,&n1,&n2,&n3);
		

		i_Tet.set_nabs(i,0,n0-1);
		i_Tet.set_nabs(i,1,n1-1);
		i_Tet.set_nabs(i,2,n2-1);
		i_Tet.set_nabs(i,3,n3-1);

		/*if(i==2093){
			printf("%d %d %d %d\n",n0,n1,n2,n3);
			printf("%d\n",i_Tet.get_nab(i,0));
		}*/
	}

	fclose(meshin);

}



//set the positon of each array by averaging the positions
//of the nodes so we can arrange the tetrahedra in a smart
//order to optimize memmory calls in the GPU

void get_tet_pos(NodeArray &i_Node,TetArray &i_Tet,int Ntets){
	int n0,n1,n2,n3;
	float xave,yave,zave;
	for (int i=0;i<Ntets;i++){
		n0 = i_Tet.get_nab(i,0);
		n1 = i_Tet.get_nab(i,1);
		n2 = i_Tet.get_nab(i,2);
		n3 = i_Tet.get_nab(i,3);

		xave = (i_Node.get_pos(n0,0) \
			   +i_Node.get_pos(n1,0) \
			   +i_Node.get_pos(n2,0) \
			   +i_Node.get_pos(n3,0))/4.0;

		yave = (i_Node.get_pos(n0,1) \
			   +i_Node.get_pos(n1,1) \
			   +i_Node.get_pos(n2,1) \
			   +i_Node.get_pos(n3,1))/4.0;

		zave = (i_Node.get_pos(n0,2) \
			   +i_Node.get_pos(n1,2) \
			   +i_Node.get_pos(n2,2) \
			   +i_Node.get_pos(n3,2))/4.0;

		i_Tet.set_pos(i,0,xave);
		i_Tet.set_pos(i,1,yave);
		i_Tet.set_pos(i,2,zave);
		i_Tet.set_pos(i,3,xave*xave+yave*yave+zave*zave);
	}
}






//re order tetrahedra so that tetrahedra which are close in number are also close
//in space so memory on GPU can be textured and accessed quicker
//use MC to minimize neighbors which are not close in memory
void gorder_tet(NodeArray &i_Node,TetArray &i_Tet,int Ntets){

	srand(98237);    //seed random number generator
	mt_init();       //initialize random number generator
	purge();         //free up memory in random number generator

	
	float dr1,dr2;
	int go=1;
	int count = 0;
	while(go==1){
		count++;
		go=0;
		for(int n1=0;n1<Ntets-1;n1++){
			dr1 = i_Tet.get_pos(n1,3);
			dr2 = i_Tet.get_pos(n1+1,3);
				if (dr2<dr1){
					go=1;
					i_Tet.switch_tets(n1,n1+1);
				}
		}//n1
	}//go==1

	float olddist,newdist;
	int n1,n2;
	float KbT = 300.0;
     count = 0;
	int tot = 0;


	//simple reordering scheme bassed only on spacial locallity
	while(KbT>0.1){
		tot++;
		count++;
		n1 = int(floor(genrand()*float(Ntets)));
		n2 = int(floor(genrand()*float(Ntets)));
		
			olddist = i_Tet.dist(n1,n1+1) \
					+ i_Tet.dist(n1,n1-1) \
					+ i_Tet.dist(n2,n2+1) \
					+ i_Tet.dist(n2,n2-1);

			newdist = i_Tet.dist(n2,n1+1) \
					+ i_Tet.dist(n2,n1-1) \
					+ i_Tet.dist(n1,n2+1) \
					+ i_Tet.dist(n1,n2-1);

			if(newdist<olddist){
				i_Tet.switch_tets(n1,n2);
				count = 0;
			}else if(genrand()<exp(-(newdist-olddist)/(KbT))){
				i_Tet.switch_tets(n1,n2);
				count = 0;
			}
		

		KbT = KbT*0.99999; //KbT*0.9999999;
		if((tot%1000)==0){
			//printf("KbT = %f count = %d\n",KbT,count);
		}
	}


	//}//count
	printf("phase 3 reordering complete\n");

	printf("tetrahedra re-orderd in %d iterations\n",tot);
	
}


//re-order nodes so that ones in tetrahedra next to each other are close
//also renumber the nodes and tetrahedra nab lists 
void finish_order(NodeArray &i_Node,TetArray &i_Tet,int Ntets, int Nnodes){

	int nrank;
	//set all new node numbers negative so we can 
	//see when one is replaced and not replace it again
	//this should account for all the redundancies
	//in the tet nab lists
	for(int i=0;i<Nnodes;i++){
		i_Node.set_newnum(i,-100);
	}
	printf("init complete\n");
	
	
	//Loop though the lists of tetrahedra and if 
	//a node has not been reassigned reassign it
	//should keep nodes in same tetrahedra close 
	//in memory and should keep nodes which share 
	//tetrahedra also close in memory
	int newi = 0;
	int i;
	for(int t = 0;t<Ntets;t++){
		for (int tn=0;tn<4;tn++){
			i = i_Tet.get_nab(t,tn);
			//printf("i = %d for t= %d and tn = %d\n",i,t,tn);
			if(i_Node.get_newnum(i)<0){
				i_Node.set_newnum(i,newi);
				newi++;
			}
		}
	}
	printf("Renumber complete newi = %d Nnodes=%d\n",newi,Nnodes);

	
	//now reassign each tetrahedra to neighbors
	//in the new arrangement of nodes
	for(int t = 0;t<Ntets;t++){
		for (int tn=0;tn<4;tn++){
			i = i_Tet.get_nab(t,tn);
			nrank = i_Node.get_totalRank(i);
			i_Tet.set_nabsRank(t,tn,nrank);
			i_Node.add_totalRank(i,1);
			i_Tet.set_nabs(t,tn,i_Node.get_newnum(i));
		}
	}
	printf("Reassign tets complete\n");

	//switch actual order of nodes
	//do this by sweeping though and switching
	//nodes which have lower real val
	//not the most efficient sort but it will work
	
	int go=1;
	while(go==1){
		go=0;
		for(int i=0;i<Nnodes-1;i++){
			if(i_Node.get_newnum(i)>i_Node.get_newnum(i+1)){
				i_Node.switch_nodes(i,i+1);
				go=1;
			}
			if(i_Node.get_newnum(i)<0){printf("nodes not properly reassigned node %d\n",i);}
		}
	}
	printf("Reordering of data complete complete\n");

	float tempVol;
	int n0,n1,n2,n3;
	for(int t=0;t<Ntets;t++){
		n0 = i_Tet.get_nab(t,0);
		n1 = i_Tet.get_nab(t,1);
		n2 = i_Tet.get_nab(t,2);
		n3 = i_Tet.get_nab(t,3);
		tempVol = tetVolume( i_Node.get_pos(n0,0)
							,i_Node.get_pos(n0,1)
							,i_Node.get_pos(n0,2)
							,i_Node.get_pos(n1,0)
							,i_Node.get_pos(n1,1)
							,i_Node.get_pos(n1,2)
							,i_Node.get_pos(n2,0)
							,i_Node.get_pos(n2,1)
							,i_Node.get_pos(n2,2)
							,i_Node.get_pos(n3,0)
							,i_Node.get_pos(n3,1)
							,i_Node.get_pos(n3,2));

		i_Tet.set_volume(t,tempVol);

	}//t
	//calculate volume of each tetrahedra



	//calculate effective volume of each node
	for(int t = 0;t<Ntets;t++){
		tempVol = 0.25*i_Tet.get_volume(t);
		for (int tn=0;tn<4;tn++){
			i = i_Tet.get_nab(t,tn);
			i_Node.add_volume(i,tempVol);
		}
	}

	//normalize volume so that each node
	//has an average volume of 1
	//i_Node.normalize_volume(float(Nnodes));

	//calculate total volume
	i_Tet.calc_total_volume();


	
}






#endif//__GETMESH_H__
