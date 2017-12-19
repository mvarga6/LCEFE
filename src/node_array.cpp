#include "node_array.h"
#include "helpers_math.h"
#include <stdio.h>

NodeArray::NodeArray(int l){
	size = l;
	MyTet = new int[size];
	MyPos = new real[size*3];
	MyForce = new real[size*3]; 
	NewNum = new int[size];
	totalRank = new int[size];
	volume = new real[size];


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

void NodeArray::reorder(std::vector<int> const &order)
{   
    for ( int s = 0, d; s < order.size(); ++s) 
    {
        for ( d = order[s]; d < s; d = order[d]);
        if (d == s) 
        {
        	while (d = order[d], d != s ) 
        	{
        		//swap( v[s], v[d]);
        		this->switch_nodes(s, d);
        	}	
        }
    }
}

void NodeArray::set_pos(int i, int j, const real &newval){	
			MyPos[i*3+j] = newval;			
}

void NodeArray::set_tet(int i, const int &newval){	
			MyTet[i] = newval;	
}


void NodeArray::add_totalRank(int i, const int &newval){	
			totalRank[i] += newval;	
			if(totalRank[i]>=MaxNodeRank){printf("Error: MaxNodeRank to low!\n");}
}

void NodeArray::set_force(int i, int j, const real &newval){	
			MyForce[i*3+j] = newval;			
}

void NodeArray::set_newnum(int i,const int &newval){
	NewNum[i] = newval;
}

real NodeArray::get_pos(int i, int j){
			return MyPos[i*3+j];			
}

int NodeArray::get_tet(int i){	
			return MyTet[i];
}

int NodeArray::get_totalRank(int i){	
			return totalRank[i];
}


real NodeArray::get_force(int i, int j){	
			return MyForce[i*3+j];			
}

int NodeArray::get_newnum(int i){
		    return NewNum[i];
}

void NodeArray::switch_nodes(int i, int j){

	real buffpos,bufftet,buffF,buffn; 
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


void NodeArray::add_volume(int i, const real &newval){
		volume[i] += newval;
}

real NodeArray::get_volume(int i){
	return volume[i];
}

void NodeArray::normalize_volume(real realVolume){
	real totalVolume = 0.0;

	for(int n=0;n<size;n++){totalVolume+=volume[n];}

	real norm = realVolume/totalVolume;
	for(int n=0;n<size;n++){volume[n]=volume[n]*norm;}
}

real NodeArray::max_point(int cord){
	real buff,max = -10000.0;

	for(int n = 0;n<size;n++){
		buff = MyPos[n*3+cord];
		if(buff>max){
			max = buff;
		}//if(buff>max)

	}//n

	return max;
}//max_point

real NodeArray::min_point(int cord){
	real buff,min = 100000.0;

	for(int n = 0;n<size;n++){
		buff = MyPos[n*3+cord];
		if(buff<min){
			min = buff;
		}//if(buff>min)

	}//n

	return min;
}//max_point

// Methods to Manipulate nodes at a whole body 
void NodeArray::translate(const real &tx = 0, const real &ty = 0, const real &tz = 0){
	for(int n = 0; n < size; n++){
		MyPos[n*3+0] += tx;
		MyPos[n*3+1] += ty;
		MyPos[n*3+2] += tz;
	}
}

// Euler Rotation in Radians preserves COM and body size
void NodeArray::eulerRotation(const real &about_z = 0, const real &about_new_x = 0, const real &about_new_z = 0){
	const real phi = about_z;
	const real the = about_new_x;
	const real psi = about_new_z;

	// centor of mass
	const real comx = 0.5f * (this->max_point(0) - this->min_point(0));
	const real comy = 0.5f * (this->max_point(1) - this->min_point(1));
	const real comz = 0.5f * (this->max_point(2) - this->min_point(2));

	// translate COM to origin
	this->translate(-comx, -comy, -comz);

	// calculate components of rotation
	const real costhe = cos(the), sinthe = sin(the),
		cospsi = cos(psi), sinpsi = sin(psi),
		cosphi = cos(phi), sinphi = sin(phi);

	// calcuate rotation matrix
	const real a11 =  cospsi*cosphi - costhe*sinphi*sinpsi,
		    a12 =  cospsi*sinphi + costhe*cosphi*sinpsi,
		    a13 =  sinpsi*sinthe,
		    a21 = -sinpsi*cosphi - costhe*sinphi*cospsi,
		    a22 = -sinpsi*sinphi + costhe*cosphi*cospsi,
		    a23 =  cospsi*sinthe,
		    a31 =  sinthe*sinphi,
		    a32 = -sinthe*cosphi,
		    a33 =  costhe;

	// apply rotation to all node positions
	real x, y, z, xp, yp, zp;
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

void NodeArray::deform(const real lambda[3]){
	real min[3], shifted_scaled;
	for(int d = 0; d < 3; d++) min[d] = this->min_point(d);
	for(int n = 0; n < size; n++){
		for(int c = 0; c < 3; c++){
			shifted_scaled = (MyPos[n*3+c] - min[c])*lambda[c] + min[c];
			//MyPos[n*3+c] *= lambda[c];
			MyPos[n*3+c] = shifted_scaled;
		}
	}
}


real NodeArray::dist(int i, int j)
{
	const real dx = MyPos[j*3 + 0] - MyPos[i*3 + 0];
	const real dy = MyPos[j*3 + 1] - MyPos[i*3 + 1];
	const real dz = MyPos[j*3 + 2] - MyPos[i*3 + 2];
	return (real)sqrt(dx*dx + dy*dy + dz*dz);
	//return math::dist(MyPos[i*3 + 0], MyPos[i*3 + 1], MyPos[i*3 + 2],
	//	MyPos[j*3 + 0], MyPos[j*3 + 1], MyPos[j*3 + 2]);
}


void NodeArray::disp(int i, int j, real r[3])
{
	r[0] = MyPos[j*3 + 0] - MyPos[i*3 + 0];
	r[1] = MyPos[j*3 + 1] - MyPos[i*3 + 1];
	r[2] = MyPos[j*3 + 2] - MyPos[i*3 + 2];
	real mag = (real)sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
	r[0] /= mag; r[1] /= mag; r[2] /= mag;
	//return math::disp(MyPos[i*3 + 0], MyPos[i*3 + 1], MyPos[i*3 + 2],
	//	MyPos[j*3 + 0], MyPos[j*3 + 1], MyPos[j*3 + 2], r);
}
