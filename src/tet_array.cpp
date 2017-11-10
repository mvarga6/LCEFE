#include "tet_array.h"
#include "parameters.h"

TetArray::TetArray(int N){
	size = N;
	TetVolume = new real[size];
	TetNab = new int[size*4];
	TetPos = new real[size*4];    ///x y z |r|
	TetA = new real[size*16];
	TetinvA = new real[size*16];
	TetNodeRank = new int[size*4];
	ThPhi = new real[size*2];
	S = new int[size];
	totalVolume = 0.0;

	for(int i=0;i<size*4;i++){
		TetNodeRank[i] =0;
		if(i<size){
			TetVolume[i] = 0.0;
			S[i] = S0*SRES; // init S to -1 for debugging
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

void TetArray::set_A(int i, int j, int k,const real &newval){
		TetA[i*16+j*4+k]=newval;
}

void TetArray::set_volume(int i, const real &newval){
		TetVolume[i] = newval;
}

real TetArray::get_volume(int i){
		return TetVolume[i];
}

real TetArray::get_A(int i, int j, int k){
		return TetA[i*16+j*4+k];
}

void TetArray::set_invA(int i, int j, int k,const real &newval){
		TetinvA[i*16+j*4+k]=newval;
}

real TetArray::get_invA(int i, int j, int k){
		return TetinvA[i*16+j*4+k];
}

void TetArray::set_nabs(int i, int j,const int &newval){
		TetNab[i*4+j] = newval;
}

void TetArray::set_nabsRank(int i, int j,const int &newval){
		TetNodeRank[i*4+j] = newval;
}

void TetArray::set_pos(int i, int j,const real &newval){
		TetPos[i*4+j] = newval;		
}

int TetArray::get_nab(int i, int j){	
		return TetNab[i*4+j];		
}

int TetArray::get_nabRank(int i, int j){	
		return TetNodeRank[i*4+j];		
}

real TetArray::get_pos(int i, int j){	
		return TetPos[i*4+j];		
}

void TetArray::set_theta(int i ,const real &newval){
		ThPhi[i*2] = newval;
}

void TetArray::set_phi(int i ,const real &newval){
		ThPhi[i*2+1] = newval;
}

// sets S for ith tet by converting to int with _S_RES factor
void TetArray::set_S(int i, const real &newval){
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

real TetArray::get_fS(int i){ //returns real
	return (real(this->S[i]) / SRES); // converts S back to real in [ 0.0 : 1.0 ]
}

int TetArray::get_iS(int i){
	return this->S[i]; //returns int w/o converting back to real range
}

void TetArray::printDirector(std::string outputBase)
{
  real th, ph;
  char fout[128];
  sprintf(fout, "%s_dir.vtk", outputBase.c_str());
  FILE * out;
//  out = fopen("Output//dir.vtk","w");
  out = fopen(fout,"w");
  fprintf(out,"# vtk DataFile Version 3.1\n");
  fprintf(out,"director profile\n");
  fprintf(out,"ASCII\n");
  fprintf(out,"DATASET UNSTRUCTURED_GRID\n");
  fprintf(out,"\n");
  fprintf(out,"POINTS %d real\n",size);
  
  //loop over tetrahedras to get positons
  for (int i = 0; i < size; i++)
  {
    fprintf(out,"%f %f %f\n",TetPos[i*4],TetPos[i*4+1],TetPos[i*4+2]);
  }//i
  fprintf(out,"\n");

  //cells
  fprintf(out,"CELLS %d %d\n",size,size*2);
  for(int i = 0; i < size; i++)
  {
    fprintf(out,"1 %d\n",i);
  }//i
  fprintf(out,"\n");
  
  //cell types
  fprintf(out,"CELL_TYPES %d\n",size);
  for(int i = 0; i < size; i++)
  {
    fprintf(out,"1\n");
  }//i
  fprintf(out,"\n");

  //vector data
  fprintf(out,"POINT_DATA %d\n",size);
  fprintf(out,"VECTORS director real\n");
  for(int i = 0; i < size; i++)
  {
    th = ThPhi[i*2];
    ph = ThPhi[i*2+1];
    fprintf(out,"%f %f %f\n",sin(th)*cos(ph),sin(th)*sin(ph),cos(th));
  }//i
  fprintf(out,"\n");

  fclose(out); 

}//print director

//switch all elemnts of both Tet arrays for i and j
void TetArray::switch_tets(int i, int j){
	real buffpos,buffA,buffTh,buffPhi;
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


real TetArray::are_we_bros(int n1, int n2){
	real ret=0.0;
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

real TetArray::dist(int n1, int n2){
real dx,dy,dz;


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
	real totalVOLUME = 0.0;
	for (int t=0;t<size;t++){
		totalVOLUME+=TetVolume[t];
	}//t
	totalVolume =  totalVOLUME;
}//calc_total_volume


real TetArray::get_total_volume(){
	return totalVolume;
}//get total volume



real TetArray::max(int cord){
	real maxVal = -100000.0;
	real tempVal;

	for (int t=0;t<size;t++){
		tempVal = TetPos[t*4+cord];
		if(tempVal>maxVal){maxVal=tempVal;}
	}//t

	return maxVal;
}//find largest point in [0 1 2] = [x y z] directions

real TetArray::min(int cord){
	real minVal = 100000.0;
	real tempVal;

	for (int t=0;t<size;t++){
		tempVal = TetPos[t*4+cord];
		if(tempVal<minVal){minVal=tempVal;}
	}//t

	return minVal;
}//find smallest point in [0 1 2] = [x y z] directions
