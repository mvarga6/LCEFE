#ifndef __DIRECTORRELAX_FUNCTIONS_H__
#define __DIRECTORRELAX_FUNCTIONS_H__

#include <math.h>

//funtions tobe used by directorRelax

void initAll(float *THETA
			,float *PHI
			,float *nx
			,float *ny
			,float *nz
			,float *tx
			,float *ty
			,float *tz
			,int xSize
			,int ySize
			,int zSize){

float theta,phi;

	//initialize everything-------------------------------------
	for(int i=0;i<xSize;i++){
		for(int j=0;j<ySize;j++){
			for(int k=0;k<zSize;k++){

				//set initial orientation
				theta = THETA[(i*ySize+ j)*zSize+k];
				phi = PHI[(i*ySize+ j)*zSize+k];

				nx[(i*ySize+ j)*zSize+k] = sin(theta)*cos(phi);
				ny[(i*ySize+ j)*zSize+k] = sin(theta)*sin(phi);
				nz[(i*ySize+ j)*zSize+k] = cos(theta);

				//zero out other arrays
				tx[(i*ySize+ j)*zSize+k] = 0.0;
				ty[(i*ySize+ j)*zSize+k] = 0.0;
				tz[(i*ySize+ j)*zSize+k] = 0.0;

			}//k
		}//j
	}//i


}//initALL


//calculate the torque on all directors
void calcTorque(float *nx
			   ,float *ny
			   ,float *nz
			   ,float *tx
			   ,float *ty
			   ,float *tz
			   ,int xSize
			   ,int ySize
			   ,int zSize){
//intialization
float dott,crossx,crossy,crossz,Tx,Ty,Tz;
int ip1,jp1,kp1,my,myip1,myjp1,mykp1;
float K = 1.0,a = 1.0;
//use open boundary conditions for now
for(int i=0;i<xSize-1;i++){
	ip1 = i+1;
		for(int j=0;j<ySize-1;j++){
			jp1 = j+1;
			for(int k=0;k<zSize-1;k++){
				kp1 = k+1;

				//R3->R1 mapping
				my = (i*ySize+ j)*zSize+k;
				myip1 = (ip1*ySize+ j)*zSize+k;
				myjp1 = (i*ySize+ jp1)*zSize+k;
				mykp1 = (i*ySize+ j)*zSize+kp1;


				//in i direction
				dott = nx[my]*nx[myip1]+ny[my]*ny[myip1]+nz[my]*nz[myip1];
				crossx = ny[my]*nz[myip1]-nz[my]*nz[myip1];
				crossy = nz[my]*nx[myip1]-nx[my]*nz[myip1];
				crossz = nx[my]*ny[myip1]-ny[my]*nx[myip1];
                Tx = 2.0*K*a*crossx*dott;
                Ty = 2.0*K*a*crossy*dott;
                Tz = 2.0*K*a*crossz*dott;
                //Tx = Tx - a*a*K*q0*(crossx*crossx + 1.0*dott*nx[my]*nx[myip1]-dott*dott);
                //Ty = Ty - a*a*K*q0*(crossy*crossx + 1.0*dott*nx[my]*ny[myip1]);
                //Tz = Tz - a*a*K*q0*(crossz*crossx + 1.0*dott*nx[my]*nz[myip1]);
                tx[my] += Tx;
                tx[myip1] += -Tx;
				ty[my] += Ty;
                ty[myip1] += -Ty;
				tz[my] += Tz;
                tz[myip1] += -Tz;

				//in j direction
				dott = nx[my]*nx[myjp1]+ny[my]*ny[myjp1]+nz[my]*nz[myjp1];
				crossx = ny[my]*nz[myjp1]-nz[my]*nz[myjp1];
				crossy = nz[my]*nx[myjp1]-nx[my]*nz[myjp1];
				crossz = nx[my]*ny[myjp1]-ny[my]*nx[myjp1];
                Tx = 2.0*K*a*crossx*dott;
                Ty = 2.0*K*a*crossy*dott;
                Tz = 2.0*K*a*crossz*dott;
                //Tx = Tx - a*a*K*q0*(crossx*crossx + 1.0*dott*nx[my]*nx[myjp1]);
                //Ty = Ty - a*a*K*q0*(crossy*crossx + 1.0*dott*nx[my]*ny[myjp1]-dott*dott);
                //Tz = Tz - a*a*K*q0*(crossz*crossx + 1.0*dott*nx[my]*nz[myjp1]);
                tx[my] += Tx;
                tx[myjp1] += -Tx;
				ty[my] += Ty;
                ty[myjp1] += -Ty;
				tz[my] += Tz;
                tz[myjp1] += -Tz;

				//in k direction
				dott = nx[my]*nx[myjp1]+ny[my]*ny[mykp1]+nz[my]*nz[mykp1];
				crossx = ny[my]*nz[mykp1]-nz[my]*nz[mykp1];
				crossy = nz[my]*nx[mykp1]-nx[my]*nz[mykp1];
				crossz = nx[my]*ny[mykp1]-ny[my]*nx[mykp1];
                Tx = 2.0*K*a*crossx*dott;
                Ty = 2.0*K*a*crossy*dott;
                Tz = 2.0*K*a*crossz*dott;
                //Tx = Tx - a*a*K*q0*(crossx*crossx + 1.0*dott*nx[my]*nx[mykp1]);
                //Ty = Ty - a*a*K*q0*(crossy*crossx + 1.0*dott*nx[my]*ny[mykp1]);
                //Tz = Tz - a*a*K*q0*(crossz*crossx + 1.0*dott*nx[my]*nz[mykp1]-dott*dott);
                tx[my] += Tx;
                tx[mykp1] += -Tx;
				ty[my] += Ty;
                ty[mykp1] += -Ty;
				tz[my] += Tz;
                tz[mykp1] += -Tz;

			}//k
		}//j
	}//i


}//calcTorque
			   


//update the director
void updateN(   float *nx
			   ,float *ny
			   ,float *nz
			   ,float *tx
			   ,float *ty
			   ,float *tz
			   ,int xSize
			   ,int ySize
			   ,int zSize
			   ,float deltat){

//intialize variabels
float nxNew,nyNew,nzNew,norm;
int my;

//top and bottom do not update
	for(int i=0;i<xSize;i++){
		for(int j=0;j<ySize;j++){
			for(int k=1;k<zSize-1;k++){

				//R3->R1 mapping
				my = (i*ySize+ j)*zSize+k;

				//calculate new director (over damped dynamics)
				nxNew = nx[my] + (tx[my]*nz[my]-tz[my]*ny[my])*deltat;
				nyNew = ny[my] + (tz[my]*nx[my]-tx[my]*nz[my])*deltat;
				nzNew = nz[my] + (tx[my]*ny[my]-ty[my]*nx[my])*deltat;
				
				//normalize new director
				norm = sqrt(nxNew*nxNew+nyNew*nyNew+nzNew*nzNew);

				//save new diretor
				nx[my] = nxNew/norm;
				ny[my] = nyNew/norm;
				nz[my] = nzNew/norm;
			
			}//k
		}//j
	}//i


}//updateN



//zero out the torque vecrors
void zeroTorque(float * tx
			   ,float * ty
			   ,float * tz
			   ,int xSize
			   ,int ySize
			   ,int zSize){

//variables
int my;

for(int i=0;i<xSize;i++){
		for(int j=0;j<ySize;j++){
			for(int k=0;k<zSize;k++){

				//R3->R1 mapping
				my = (i*ySize+ j)*zSize+k;

				tx[my] = 0.0;
				ty[my] = 0.0;
				tz[my] = 0.0;

			}//k
		}//j
	}//i

}//zeroTorque


//convert nx, ny and nz back to THETA and PHI
void n2ThetaPhi(float * nx
			   ,float * ny
			   ,float * nz
			   ,float * THETA
			   ,float * PHI
			   ,int xSize
			   ,int ySize
			   ,int zSize){


//variables
int my;

   for(int i=0;i<xSize;i++){
		for(int j=0;j<ySize;j++){
			for(int k=0;k<zSize;k++){

				//R3->R1 mapping
				my = (i*ySize+ j)*zSize+k;

				//get theta and phi
				THETA[my] = acos(nz[my]);
				PHI[my] = atan2(ny[my],nx[my]);

			}//k
		}//j
	}//i

}//n2ThetaPhi

#endif//__DIRECTORRELAX_FUNCTIONS_H__