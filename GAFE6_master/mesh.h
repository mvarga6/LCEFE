#pragma once

#include "mainhead.h"
#include "loadmesh.h"

class DeviceController;

class Mesh {
	NodeArray * nodeArray;
	TetArray * tetArray;
	int Nnodes;
	int Ntets;
	friend class DeviceController;
public:
	Mesh() : nodeArray(NULL), tetArray(NULL){};
	~Mesh(){};
	// ----------------------------------------------------------------------------
	inline bool loadMeshDim(){
		get_mesh_dim(this->Ntets, this->Nnodes);
		return true;
	}
	// ----------------------------------------------------------------------------
	inline void createTetAndNodeArrays(){
		this->nodeArray = new NodeArray(this->Nnodes);
		this->tetArray = new TetArray(this->Ntets);
	}
	// ----------------------------------------------------------------------------
	inline bool loadMesh(){
		get_mesh(this->nodeArray, this->tetArray, this->Ntets, this->Nnodes);
		return true;
	}
	// ----------------------------------------------------------------------------
	// Must call 'calculateTetPositions' afterwards
	inline void rescaleMesh(const float &scaleFactor, AXIS dimension){
		for (int i = 0; i < this->Nnodes; i++){
			float scaledPos = scaleFactor * this->nodeArray->get_pos(i, dimension);
			this->nodeArray->set_pos(i, dimension, scaledPos);
		}
	}
	// ----------------------------------------------------------------------------
	inline void calculateTetPositions(){
		get_tet_pos(this->nodeArray, this->tetArray, this->Ntets);
	}
	// ----------------------------------------------------------------------------
	inline void loadDirector(){
		set_n(this->tetArray, this->Ntets);
	}
	// ----------------------------------------------------------------------------
	inline void orderTetAndNodeArrays(){
		gorder_tet(this->nodeArray, this->tetArray, this->Ntets);
		finish_order(this->nodeArray, this->tetArray, this->Ntets, this->Nnodes);
	}
	// ----------------------------------------------------------------------------
	inline void calculateAMatrices(){
		init_As(this->nodeArray, this->tetArray, this->Ntets);
	}
	// ----------------------------------------------------------------------------
	inline bool printOrderAndDirector(){
		printorder(this->tetArray, this->Ntets);
		this->tetArray->printDirector();
		return true;
	}
	// ----------------------------------------------------------------------------
};