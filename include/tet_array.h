#ifndef __TET_ARRAY_H__
#define __TET_ARRAY_H__

#include <string>
#include "defines.h"
#include <vector>

//  class to hold the tetrahedral array with instances which will be usefull for renumbering
class TetArray
{

public:
	int *TetNab;
	int *TetNodeRank;
	real *TetPos;
	real *TetA;
	real *TetinvA;
	real *TetVolume;
	real *ThPhi; // orientation of director in 3D
	real *S; // value of order parameter inside tet.  map(0 -> SRES == 0.0 -> 1.0)
	real totalVolume;
	int size;

	TetArray(const int N, const real S0 = 0);
	~TetArray();

	void set_A(int i, int j, int k,const real &newval);
	real get_A(int i, int j, int k);
	void set_invA(int i, int j, int k,const real &newval);
	real get_invA(int i, int j, int k);
	int get_size(){return size;}
	void set_nabs(int i, int j,const int &newval);
	void set_nabsRank(int i, int j,const int &newval);
	void set_pos(int i, int j,const real &newval);
	void set_volume(int i, const real &newval);
	real get_volume(int i);
	int get_nab(int i, int j);
	int get_nabRank(int i, int j);
	real get_pos(int i, int j);
	void switch_tets(int i, int j);
	real are_we_bros(int n1, int n2);
	real dist(int n1,int n2);
	void calc_total_volume();
	real get_total_volume();
	void set_theta(int i, const real &newval);
	void set_phi(int i, const real &newval);
	void set_S(int i, const real &newval);
	int get_ThPhi(int i);
	real get_S(int i);
	real max(int cord);
	real min(int cord);
	void printDirector(std::string);
	  
	///
	/// Method to reorder the tets
	void reorder(std::vector<int> const &order);
};

#endif
