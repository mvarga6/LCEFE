#ifndef __CLASSSTRUCT_H__
#define __CLASSSTRUCT_H__

#include "parameters.h"

// class to hold the information at each node
class NodeArray{

public:
	int *MyTet;
	float *MyPos;
	float *MyForce;
	int *NewNum;
	int *totalRank;
	float *volume;
	int size;



	NodeArray(int l);
	~NodeArray();

	int get_size(){return size;}
	void set_pos(int i, int j,const float &newval);
	void set_tet(int i,const int &newval);
	void set_force(int i, int j,const float &newval);
	void set_newnum(int i,const int &newval);
	void add_totalRank(int i, const int &newval);
	int get_totalRank(int i);
	float get_pos(int i, int j);
	int get_tet(int i);
	float get_force(int i, int j);
	int get_newnum(int i);
	void switch_nodes(int i, int j);
	void add_volume(int i, const float &newval);
	float get_volume(int i);
	void normalize_volume(float realVolume);
	float max_point(int cord);
	float min_point(int cord);

	// Method to manipulate nodes as whole body
	void translate(const float&, const float &, const float &);
	void eulerRotation(const float&, const float&, const float&);
	void deform(const float lambda[3]);
};


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
	for(int n = 0; n < size; n++){
		for(int c = 0; c < 3; c++){
			MyPos[n*3+c] *= lambda[c];
		}
	}
}


//  class to hold the tetrahedral array with instances which will be usefull for renumbering
class TetArray{

public:
	int *TetNab;
	int *TetNodeRank;
	float *TetPos;
	float *TetA;
	float *TetinvA;
	float *TetVolume;
	float *ThPhi; // orientation of director in 3D
	int *S; // value of order parameter inside tet.  map(0 -> SRES == 0.0 -> 1.0)
	float totalVolume;
	int size;

	TetArray(int size);
	~TetArray();

	void set_A(int i, int j, int k,const float &newval);
	float get_A(int i, int j, int k);
	void set_invA(int i, int j, int k,const float &newval);
	float get_invA(int i, int j, int k);
	int get_size(){return size;}
	void set_nabs(int i, int j,const int &newval);
	void set_nabsRank(int i, int j,const int &newval);
	void set_pos(int i, int j,const float &newval);
	void set_volume(int i, const float &newval);
	float get_volume(int i);
	int get_nab(int i, int j);
	int get_nabRank(int i, int j);
	float get_pos(int i, int j);
	void switch_tets(int i, int j);
	float are_we_bros(int n1, int n2);
	float dist(int n1,int n2);
	void calc_total_volume();
	float get_total_volume();
	void set_theta(int i, const float &newval);
	void set_phi(int i, const float &newval);
	void set_S(int i, const float &newval);
	int get_ThPhi(int i);
	float get_fS(int i);
	int get_iS(int i);
	float max(int cord);
	float min(int cord);
  	void printDirector();
};

TetArray::TetArray(int N){
	size = N;
	TetVolume = new float[size];
	TetNab = new int[size*4];
	TetPos = new float[size*4];    ///x y z |r|
	TetA = new float[size*16];
	TetinvA = new float[size*16];
	TetNodeRank = new int[size*4];
	ThPhi = new float[size*2];
	S = new int[size];
	totalVolume = 0.0;

	for(int i=0;i<size*4;i++){
		TetNodeRank[i] =0;
		if(i<size){
			TetVolume[i] = 0.0;
			S[i] = 0; // init S to -1 for debugging
		}//if i
	}//i
}

TetArray::~TetArray(){
	delete [] TetNab;
	TetNab = NULL;
	delete [] TetPos;
	TetNab = NULL;
	delete [] TetA;
	TetA= NULL;
	delete [] TetinvA;
	TetinvA= NULL;
	delete [] ThPhi;
	ThPhi = NULL;
	delete [] S;
	S = NULL;
	delete [] TetVolume;
	TetVolume = NULL;
	delete [] TetNodeRank;
	TetNodeRank = NULL;
	
}

void TetArray::set_A(int i, int j, int k,const float &newval){
		TetA[i*16+j*4+k]=newval;
}

void TetArray::set_volume(int i, const float &newval){
		TetVolume[i] = newval;
}

float TetArray::get_volume(int i){
		return TetVolume[i];
}

float TetArray::get_A(int i, int j, int k){
		return TetA[i*16+j*4+k];
}

void TetArray::set_invA(int i, int j, int k,const float &newval){
		TetinvA[i*16+j*4+k]=newval;
}

float TetArray::get_invA(int i, int j, int k){
		return TetinvA[i*16+j*4+k];
}

void TetArray::set_nabs(int i, int j,const int &newval){
		TetNab[i*4+j] = newval;
}

void TetArray::set_nabsRank(int i, int j,const int &newval){
		TetNodeRank[i*4+j] = newval;
}

void TetArray::set_pos(int i, int j,const float &newval){
		TetPos[i*4+j] = newval;		
}

int TetArray::get_nab(int i, int j){	
		return TetNab[i*4+j];		
}

int TetArray::get_nabRank(int i, int j){	
		return TetNodeRank[i*4+j];		
}

float TetArray::get_pos(int i, int j){	
		return TetPos[i*4+j];		
}

void TetArray::set_theta(int i ,const float &newval){
		ThPhi[i*2] = newval;
}

void TetArray::set_phi(int i ,const float &newval){
		ThPhi[i*2+1] = newval;
}

// sets S for ith tet by converting to int with _S_RES factor
void TetArray::set_S(int i, const float &newval){
		int ival;
		if(newval > 1.0f) ival = 1;
		else if(newval < 0) ival = 0;
		else ival = int(newval * SRES);
		this->S[i] = ival;
}

int TetArray::get_ThPhi(int i){
	int th = int(floor(1000.0*ThPhi[i*2]/PI));
	int phi = int(floor(500.0*ThPhi[i*2+1]/PI));
	return th*10000+phi;
}

float TetArray::get_fS(int i){ //returns float
	return (float(this->S[i]) / SRES); // converts S back to float in [ 0.0 : 1.0 ]
}

int TetArray::get_iS(int i){
	return this->S[i]; //returns int w/o converting back to float range
}

void TetArray::printDirector(){
  float th, ph;
  FILE*out;
  out = fopen("Output//dir.vtk","w");
  fprintf(out,"# vtk DataFile Version 3.1\n");
  fprintf(out,"director profile\n");
  fprintf(out,"ASCII\n");
  fprintf(out,"DATASET UNSTRUCTURED_GRID\n");
  fprintf(out,"\n");
  fprintf(out,"POINTS %d FLOAT\n",size);
  
  //loop over tetrahedras to get positons
  for(int i=0;i<size;i++){
    fprintf(out,"%f %f %f\n",TetPos[i*4],TetPos[i*4+1],TetPos[i*4+2]);
  }//i
  fprintf(out,"\n");

  //cells
  fprintf(out,"CELLS %d %d\n",size,size*2);
  for(int i=0;i<size;i++){
    fprintf(out,"1 %d\n",i);
  }//i
  fprintf(out,"\n");
  
  //cell types
  fprintf(out,"CELL_TYPES %d\n",size);
  for(int i=0;i<size;i++){
    fprintf(out,"1\n");
  }//i
  fprintf(out,"\n");

  //vector data
  fprintf(out,"POINT_DATA %d\n",size);
  fprintf(out,"VECTORS director FLOAT\n");
  for(int i=0;i<size;i++){
    th = ThPhi[i*2];
    ph = ThPhi[i*2+1];
    fprintf(out,"%f %f %f\n",sin(th)*cos(ph),sin(th)*sin(ph),cos(th));
  }//i
  fprintf(out,"\n");

  fclose(out); 

}//print director

//switch all elemnts of both Tet arrays for i and j
void TetArray::switch_tets(int i, int j){
	float buffpos,buffA,buffTh,buffPhi;
	int buffnab;
	if(i>-1&&i<size){
		if(j>-1&&j<size){

			buffTh = ThPhi[i*2];
			buffPhi = ThPhi[i*2+1];
			ThPhi[i*2] = ThPhi[j*2];
			ThPhi[i*2+1] = ThPhi[j*2+1];
			ThPhi[j*2] = buffTh;
			ThPhi[j*2+1] = buffPhi;

			for(int p = 0;p<16;p++){
				if(p<4){
				

				buffpos = TetPos[i*4+p];
				TetPos[i*4+p] = TetPos[j*4+p];
				TetPos[j*4+p] = buffpos;
				buffnab = TetNab[i*4+p];
				TetNab[i*4+p] = TetNab[j*4+p];
				TetNab[j*4+p] = buffnab;
				}
				buffA = TetA[i*16+p];
				TetA[i*16+p] = TetA[j*16+p];
				TetA[j*16+p] = buffA;
			}
		}
	}
}


float TetArray::are_we_bros(int n1, int n2){
	float ret=0.0;
	if(n2>-1&&n2<size){
	for (int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			if(TetNab[4*n1+i]==TetNab[4*n2+j]){
				ret += 1.0;
			}//if
		}//j
	}//i
	return ret;
	}else{
	return 0.0;
	}
}

float TetArray::dist(int n1, int n2){
float dx,dy,dz;


	if(n2>-1||n2<size){
	dx = TetPos[n1*4+0]-TetPos[n2*4+0];
	dy = TetPos[n1*4+1]-TetPos[n2*4+1];
	dz = TetPos[n1*4+2]-TetPos[n2*4+2];
	return dx*dx+dy*dy+dz*dz;
	}else{
	return 0.0;
	}
}

void TetArray::calc_total_volume(){
	float totalVOLUME = 0.0;
	for (int t=0;t<size;t++){
		totalVOLUME+=TetVolume[t];
	}//t
	totalVolume =  totalVOLUME;
}//calc_total_volume


float TetArray::get_total_volume(){
	return totalVolume;
}//get total volume



float TetArray::max(int cord){
	float maxVal = -100000.0;
	float tempVal;

	for (int t=0;t<size;t++){
		tempVal = TetPos[t*4+cord];
		if(tempVal>maxVal){maxVal=tempVal;}
	}//t

	return maxVal;
}//find largest point in [0 1 2] = [x y z] directions

float TetArray::min(int cord){
	float minVal = 100000.0;
	float tempVal;

	for (int t=0;t<size;t++){
		tempVal = TetPos[t*4+cord];
		if(tempVal<minVal){minVal=tempVal;}
	}//t

	return minVal;
}//find smallest point in [0 1 2] = [x y z] directions

#endif //__DCLASSSTRUCT_H__
