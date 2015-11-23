#pragma once

#include "mainhead.h"
#include "loadmesh.h"

class DeviceController;

/* -------------------------------------------------------------------------------
	Object contains all information about a physical body made
	from nodes connected into a mesh.  Consists of a node 
	container, storing all the points in the mesh, and a 
	tetrahedral container, storing all information about the
	connections of nodes into groups of four (tetrahedral) 
	which file the space of the elastic body.
---------------------------------------------------------------------------------*/
class Mesh {

	//.. container for nodes
	NodeArray * nodeArray;
	int Nnodes;

	//.. container for tetrahedrals
	TetArray * tetArray;
	int Ntets;

	//.. DeviceController needs access
	friend class DeviceController;

public:

	Mesh() : nodeArray(NULL), tetArray(NULL){};
	~Mesh(){};
	// ----------------------------------------------------------------------------
	// grabs mesh dimensions from MESHFILE
	inline bool loadMeshDim(){
		get_mesh_dim(this->Ntets, this->Nnodes);
		return true;
	}
	// ----------------------------------------------------------------------------
	// allocates node and tet objects
	inline void createTetAndNodeArrays(){
		this->nodeArray = new NodeArray(this->Nnodes);
		this->tetArray = new TetArray(this->Ntets);
	}
	// ----------------------------------------------------------------------------
	// reads in node and tets from MESHFILE
	inline bool loadMesh(){
		get_mesh(this->nodeArray, this->tetArray, this->Ntets, this->Nnodes);
		return true;
	}
	// ----------------------------------------------------------------------------
	// rescales currently load mesh positions.
	// must call 'void calculateTetPositions()' afterwards.
	inline void rescaleMesh(const float &scaleFactor, AXIS dimension){
		for (int i = 0; i < this->Nnodes; i++){
			float scaledPos = scaleFactor * this->nodeArray->get_pos(i, dimension);
			this->nodeArray->set_pos(i, dimension, scaledPos);
		}
	}
	// ----------------------------------------------------------------------------
	// calculates and sets tet positions based on average of nodes
	inline void calculateTetPositions(){
		get_tet_pos(this->nodeArray, this->tetArray, this->Ntets);
	}
	// ----------------------------------------------------------------------------
	// sets the director for volume in each tet
	inline void loadDirector(){
		set_n(this->tetArray, this->Ntets);
	}
	// ----------------------------------------------------------------------------
	// reorders arrays for texturing data on the GPU
	inline void orderTetAndNodeArrays(){
		gorder_tet(this->nodeArray, this->tetArray, this->Ntets);
		finish_order(this->nodeArray, this->tetArray, this->Ntets, this->Nnodes);
	}
	// ----------------------------------------------------------------------------
	// calculates and sets A matrices used for force calculation
	inline void calculateAMatrices(){
		init_As(this->nodeArray, this->tetArray, this->Ntets);
	}
	// ----------------------------------------------------------------------------
	// print director in host data to files
	inline bool printOrderAndDirector(){
		return this->tetArray->printDirectorXYZV("mv_dir2");
		//printorder(this->tetArray, this->Ntets);
		//this->tetArray->printDirector();
		//return true;
	}
	// ----------------------------------------------------------------------------
};