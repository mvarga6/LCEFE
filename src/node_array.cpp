#include "node_array.h"
#include <stdio.h>

NodeArray::NodeArray(int l){
	size = l;
	MyTet = new int[size];
	MyPos = new float[size*3];
	MyForce = new float[size*3]; 
	NewNum = new int[size];
	totalRank = new int[size];
	volume = new float[size];


	for(int i=0;i<size;i++){
		volume[i] = 0.0;
		totalRank[i]=0;
		MyForce[3*i] = 0.0;
		MyForce[3*i+1] = 0.0;
		MyForce[3*i+2] = 0.0;
	}
}

NodeArray::~NodeArray(){
	delete [] MyTet;
	MyTet = NULL;
	delete [] MyPos;
	MyPos = NULL;
	delete [] MyForce; 
	MyForce = NULL;
	delete [] NewNum;
	NewNum = NULL;
	delete [] totalRank;
	totalRank = NULL;
	delete [] volume;
	volume = NULL;
}

void NodeArray::set_pos(int i, int j, const float &newval){	
			MyPos[i*3+j] = newval;			
}

void NodeArray::set_tet(int i, const int &newval){	
			MyTet[i] = newval;	
}


void NodeArray::add_totalRank(int i, const int &newval){	
			totalRank[i] += newval;	
			if(totalRank[i]>=MaxNodeRank){printf("Error: MaxNodeRank to low!\n");}
}

void NodeArray::set_force(int i, int j, const float &newval){	
			MyForce[i*3+j] = newval;			
}

void NodeArray::set_newnum(int i,const int &newval){
	NewNum[i] = newval;
}

float NodeArray::get_pos(int i, int j){
			return MyPos[i*3+j];			
}

int NodeArray::get_tet(int i){	
			return MyTet[i];
}

int NodeArray::get_totalRank(int i){	
			return totalRank[i];
}


float NodeArray::get_force(int i, int j){	
			return MyForce[i*3+j];			
}

int NodeArray::get_newnum(int i){
		    return NewNum[i];
}

void NodeArray::switch_nodes(int i, int j){

	float buffpos,bufftet,buffF,buffn; 
	int bufftrank;

		    bufftrank= totalRank[i];
			totalRank[i] = totalRank[j];
			totalRank[j] = bufftrank;
			buffn = NewNum[i];
			NewNum[i] = NewNum[j];
			NewNum[j] = buffn;
			for(int p = 0;p<3;p++){
				
				//switch force and position
					buffF = MyForce[i*3+p];
					MyForce[i*3+p] = MyForce[j*3+p];
					MyForce[j*3+p] = buffF;

					buffpos = MyPos[i*3+p];
					MyPos[i*3+p] = MyPos[j*3+p];
					MyPos[j*3+p] = buffpos;

					if(p<1){//switch mytet
					bufftet = MyTet[i];
					MyTet[i] = MyTet[j];
					MyTet[j] = bufftet;
					}
			}	
}


void NodeArray::add_volume(int i, const float &newval){
		volume[i] += newval;
}

float NodeArray::get_volume(int i){
	return volume[i];
}

void NodeArray::normalize_volume(float realVolume){
	float totalVolume = 0.0;

	for(int n=0;n<size;n++){totalVolume+=volume[n];}

	float norm = realVolume/totalVolume;
	for(int n=0;n<size;n++){volume[n]=volume[n]*norm;}
}

float NodeArray::max_point(int cord){
	float buff,max = -10000.0;

	for(int n = 0;n<size;n++){
		buff = MyPos[n*3+cord];
		if(buff>max){
			max = buff;
		}//if(buff>max)

	}//n

	return max;
}//max_point

float NodeArray::min_point(int cord){
	float buff,min = 100000.0;

	for(int n = 0;n<size;n++){
		buff = MyPos[n*3+cord];
		if(buff<min){
			min = buff;
		}//if(buff>min)

	}//n

	return min;
}//max_point

// Methods to Manipulate nodes at a whole body 
void NodeArray::translate(const float &tx = 0, const float &ty = 0, const float &tz = 0){
	for(int n = 0; n < size; n++){
		MyPos[n*3+0] += tx;
		MyPos[n*3+1] += ty;
		MyPos[n*3+2] += tz;
	}
}

// Euler Rotation in Radians preserves COM and body size
void NodeArray::eulerRotation(const float &about_z = 0, const float &about_new_x = 0, const float &about_new_z = 0){
	const float phi = about_z;
	const float the = about_new_x;
	const float psi = about_new_z;

	// centor of mass
	const float comx = 0.5f * (this->max_point(0) - this->min_point(0));
	const float comy = 0.5f * (this->max_point(1) - this->min_point(1));
	const float comz = 0.5f * (this->max_point(2) - this->min_point(2));

	// translate COM to origin
	this->translate(-comx, -comy, -comz);

	// calculate components of rotation
	const float costhe = cos(the), sinthe = sin(the),
		cospsi = cos(psi), sinpsi = sin(psi),
		cosphi = cos(phi), sinphi = sin(phi);

	// calcuate rotation matrix
	const float a11 =  cospsi*cosphi - costhe*sinphi*sinpsi,
		    a12 =  cospsi*sinphi + costhe*cosphi*sinpsi,
		    a13 =  sinpsi*sinthe,
		    a21 = -sinpsi*cosphi - costhe*sinphi*cospsi,
		    a22 = -sinpsi*sinphi + costhe*cosphi*cospsi,
		    a23 =  cospsi*sinthe,
		    a31 =  sinthe*sinphi,
		    a32 = -sinthe*cosphi,
		    a33 =  costhe;

	// apply rotation to all node positions
	float x, y, z, xp, yp, zp;
	for(int n = 0; n < size; n++){
		x = MyPos[n*3+0];
		y = MyPos[n*3+1];
		z = MyPos[n*3+2];

		xp = a11*x + a12*y + a13*z;
		yp = a21*x + a22*y + a23*z;
		zp = a31*x + a32*y + a33*z;

		MyPos[n*3+0] = xp;
		MyPos[n*3+1] = yp;
		MyPos[n*3+2] = zp;
	}

	// translate COM back to initial position
	this->translate(comx, comy, comz);
}

void NodeArray::deform(const float lambda[3]){
	float min[3], shifted_scaled;
	for(int d = 0; d < 3; d++) min[d] = this->min_point(d);
	for(int n = 0; n < size; n++){
		for(int c = 0; c < 3; c++){
			shifted_scaled = (MyPos[n*3+c] - min[c])*lambda[c] + min[c];
			//MyPos[n*3+c] *= lambda[c];
			MyPos[n*3+c] = shifted_scaled;
		}
	}
}
