#include "tet_array.h"
#include "parameters.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

static const int PrintRankIdx = 14560;

TetArray::TetArray(const int N, const real S0){
	size = N;
	TetVolume = new real[size];
	TetNab = new int[size*4];
	TetPos = new real[size*4];    ///x y z |r|
	TetA = new real[size*16];
	TetinvA = new real[size*16];
	TetNodeRank = new int[size*4];
	ThPhi = new real[size*2];
	S = new real[size];
	totalVolume = 0.0;

	for (int i = 0; i < size*4; i++)
	{
		TetNodeRank[i] = 0;
		if (i < size)
		{
			TetVolume[i] = 0.0;
			S[i] = S0;
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

void TetArray::reorder(std::vector<int> const &order)
{   
	int _size = order.size();
    if (_size != this->size)
        throw std::runtime_error("The size of the new proposed order does not match the current size.");

    for ( int s = 0, d; s < _size; ++s) 
    {
        for ( d = order[s]; d < s; d = order[d]);
        if (d == s) 
        {
        	while (d = order[d], d != s ) 
        	{
        		//swap( v[s], v[d]);
        		this->switch_tets(s, d);
        	}	
        }
    }
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
		if(newval > 1.0) ival = 1;
		else if(newval < -0.5) ival = -0.5;
		else ival = newval;
		this->S[i] = ival;
}

int TetArray::get_ThPhi(int i){
	int th = int(floor(1000.0*ThPhi[i*2]/PI));
	int phi = int(floor(500.0*ThPhi[i*2+1]/PI));
	return th*10000+phi;
}

real TetArray::get_S(int i){ //returns real
	return this->S[i];
}

void TetArray::print_ranks()
{
	const int idx = PrintRankIdx;	
	printf("\n[ DEBUG ] tet: %d ranks: %d %d %d %d",
		idx,
		TetNodeRank[idx*4 + 0],
		TetNodeRank[idx*4 + 1],
		TetNodeRank[idx*4 + 2],
		TetNodeRank[idx*4 + 3]
	);
}

void TetArray::printDirector(std::string outputBase)
{
	std::string fileName(outputBase + "_dir.xyzv");

	std::ofstream fout(fileName.c_str());
	if (!fout.is_open())
	{
		printf("\n[ Error ] Failed to open director print file: %s", outputBase.c_str());
	}

	
	// variables
	real nx, ny, nz, th, ph, x, y, z;
	
	// write file header
	printf("\n[ INFO ] Writing director file: %s", fileName.c_str());
	fout << this->size << std::endl << "LCE director" << std::endl;
	for(int i = 0; i < size; i++)
	{
		th = ThPhi[i*2];
    	ph = ThPhi[i*2+1];
	
		nx = sinf(th)*cosf(ph);
    	ny = sinf(th)*sinf(ph);
    	nz = cosf(th);
    	
    	x = this->get_pos(i, 0);
    	y = this->get_pos(i, 1);
    	z = this->get_pos(i, 2);
    	
    	fout << "A " << x << " " << y << " " << z << " "
    		 << nx << " " << ny << " " << nz << std::endl;
	}
	fout.close();
	printf("\n[ INFO ] Complete");
	
//  real th, ph;
//  char fout[128];
//  sprintf(fout, "%s_dir.vtk", outputBase.c_str());
//  FILE * out;
////  out = fopen("Output//dir.vtk","w");
//  out = fopen(fout,"w");
//  fprintf(out,"# vtk DataFile Version 3.1\n");
//  fprintf(out,"director profile\n");
//  fprintf(out,"ASCII\n");
//  fprintf(out,"DATASET UNSTRUCTURED_GRID\n");
//  fprintf(out,"\n");
//  fprintf(out,"POINTS %d float\n",size);
//  
//  //loop over tetrahedras to get positons
//  for (int i = 0; i < size; i++)
//  {
//    fprintf(out,"%f %f %f\n",TetPos[i*4],TetPos[i*4+1],TetPos[i*4+2]);
//  }//i
//  fprintf(out,"\n");

//  //cells
//  fprintf(out,"CELLS %d %d\n",size,size*2);
//  for(int i = 0; i < size; i++)
//  {
//    fprintf(out,"1 %d\n",i);
//  }//i
//  fprintf(out,"\n");
//  
//  //cell types
//  fprintf(out,"CELL_TYPES %d\n",size);
//  for(int i = 0; i < size; i++)
//  {
//    fprintf(out,"1\n");
//  }//i
//  fprintf(out,"\n");

//  //vector data
//  fprintf(out,"POINT_DATA %d\n",size);
//  fprintf(out,"VECTORS director float\n");
//  real nx, ny, nz;
//  for(int i = 0; i < size; i++)
//  {
//    th = ThPhi[i*2];
//    ph = ThPhi[i*2+1];
//    nx = sinf(th)*cosf(ph);
//    ny = sinf(th)*sinf(ph);
//    nz = cosf(th);
//    fprintf(out,"%f %f %f\n", nx, ny, nz);
//  }//i
//  fprintf(out,"\n");

//  fclose(out); 

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
	for (int t = 0; t < size; t++){
		totalVOLUME += TetVolume[t];
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
