#ifndef __NODE_ARRAY_H__
#define __NODE_ARRAY_H__

#include "defines.h"
#include "parameters.h"
#include <vector>

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
	int *rankInTris;
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
	/// Increment rank of ith node for tets its in
	void increment_rank_wrt_tets(int i);

	///
	/// Increment rank of ith node for tris its in
	void increment_rank_wrt_tris(int i);

	///
	/// returns the rank of ith node w.r.t. tetrahedra
	/// that contain it
	int get_rank_wrt_tets(int i);

	///
	/// returns the rank of ith node w.r.t. triangles
	/// that contain it
	int get_rank_wrt_tris(int i);

	///
	/// Method to translate the nodes in Euclidean space
	void translate(const real&, const real &, const real &);

	///
	/// Euler rotation of the nodes
	void eulerRotation(const real&, const real&, const real&);

	///
	/// Deform the nodes positions in x,y,z
	void deform(const real lambda[3]);

	///
	/// Method to reorder the nodes
	void reorder(std::vector<int> const &order);

	///
	/// Distance between two nodes
	real dist(int i, int j);

	///
	/// Displacement vector between two nodes
	void disp(int i, int j, real r[3]);

};

#endif
