#include "mesh.h"
#include "getgmsh.h"
#include "getmesh.h"

Mesh::Mesh(SimulationParameters *parameters)
{
	this->loaded = false;
	this->params = parameters;
}


bool Mesh::Load()
{
	//get dimensions of the mesh
	MeshDimensions meshDim;
	try
	{
		meshDim = get_gmsh_dim(this->params->Mesh.File);
	}
	catch (const std::exception& e)
	{
		// print something
		return false;
	}
	
	// unload if loaded
	if (loaded)
	{
		delete this->Tets;
		delete this->Nodes;
	}
	
	// allocate the containers for nodes and tets
	this->Tets = new TetArray(meshDim.Ntets);
	this->Nodes = new NodeArray(meshDim.Nnodes);	
	
	// read the positions of nodes and tet indices
	try
	{
		get_gmsh(this->params->Mesh.File, *Nodes, *Tets, this->params->Mesh.Scale);
	}
	catch (const std::exception& e)
	{
		// print something
		return false;
	}
	
	// read the positions of nodes and tet indices
	try
	{
		get_tet_pos(*Nodes, *Tets);
	}
	catch (const std::exception& e)
	{
		// print something
		return false;
	}
	
	loaded = true;
	return true;
}



void Mesh::SetDirector(DirectorField *field)
{
	int Ntets = this->Tets->size;
	
	DirectorOrientation dir;
	float x, y, z;
	
	for (int t = 0; t < Ntets; t++)
	{
		// the position of this tet
		x = this->Tets->get_pos(t, 0);
		y = this->Tets->get_pos(t, 0);
		z = this->Tets->get_pos(t, 0);
	
		// get the director there
		// done this way so it could be read
		// from file or hardcoded etc
		dir = field->GetDirectorAt(x, y, z);
		
		// assign 
		this->Tets->set_theta(t, dir.theta);
		this->Tets->set_phi(t, dir.phi);
	}
}



void Mesh::Apply(MeshOptimizer *optimizer)
{
	OptimizationResult result;
	result = optimizer->Run(this->Tets, this->Nodes);
	
	switch(result)
	{
	case OptimizationResult::SUCCESS:
		// log success
		return;
		
	case OptimizationResult::FAILURE_NO_OPTIMATION:
		// log no optimization
		return;
		
	case OptimizationResult::FAILURE_EXCEPTION_THROWN:
		// log exception
		exit(101);
		
	default:
		// log unknown result
		return;
	}
}



void Mesh::Update()
{
	
}
