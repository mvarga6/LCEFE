#ifndef __NODE_ARRAY_H__
#define __NODE_ARRAY_H__

#include "parameters.h"

class NodeArray
{

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

#endif
