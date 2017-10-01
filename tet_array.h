#ifndef __TET_ARRAY_H__
#define __TET_ARRAY_H__

#include <string>

//  class to hold the tetrahedral array with instances which will be usefull for renumbering
class TetArray
{

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
  	void printDirector(std::string);
};

#endif
