#ifndef __NODE_ARRAY_H__
#define __NODE_ARRAY_H__

#include "defines.h"
#include "parameters.h"

///
/// Host-only constainer for node data
class NodeArray
{

public:
	int *MyTet;
	real *MyPos;
	real *MyForce;
	int *NewNum;
	int *totalRank;
	real *volume;
	int size;



	NodeArray(int l);
	~NodeArray();

	int get_size(){return size;}
	void set_pos(int i, int j,const real &newval);
	void set_tet(int i,const int &newval);
	void set_force(int i, int j,const real &newval);
	void set_newnum(int i,const int &newval);
	void add_totalRank(int i, const int &newval);
	int get_totalRank(int i);
	real get_pos(int i, int j);
	int get_tet(int i);
	real get_force(int i, int j);
	int get_newnum(int i);
	void switch_nodes(int i, int j);
	void add_volume(int i, const real &newval);
	real get_volume(int i);
	void normalize_volume(real realVolume);
	real max_point(int cord);
	real min_point(int cord);

	///
	/// Method to translate the nodes in Euclidean space
	void translate(const real&, const real &, const real &);

	///
	/// Euler rotation of the nodes
	void eulerRotation(const real&, const real&, const real&);

	///
	/// Deform the nodes positions in x,y,z
	void deform(const real lambda[3]);
};

#endif
