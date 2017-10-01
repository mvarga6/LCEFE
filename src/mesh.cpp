#include "mesh.h"
#include "getgmsh.h"
#include "getmesh.h"
#include "getAs.h"

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
	try
	{
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
	catch (const std::exception& e)
	{
		// print something
		return;
	}
}



bool Mesh::CalculateVolumes()
{
	try
	{
		float tempVol;
		int n0, n1, n2, n3;
		int Ntets = this->Tets->size;
	
		//calculate volume of each tetrahedra
		for(int t = 0;t < Ntets; t++)
		{
			n0 = Tets->get_nab(t,0);
			n1 = Tets->get_nab(t,1);
			n2 = Tets->get_nab(t,2);
			n3 = Tets->get_nab(t,3);
			tempVol = tetVolume( Nodes->get_pos(n0,0)
								,Nodes->get_pos(n0,1)
								,Nodes->get_pos(n0,2)
								,Nodes->get_pos(n1,0)
								,Nodes->get_pos(n1,1)
								,Nodes->get_pos(n1,2)
								,Nodes->get_pos(n2,0)
								,Nodes->get_pos(n2,1)
								,Nodes->get_pos(n2,2)
								,Nodes->get_pos(n3,0)
								,Nodes->get_pos(n3,1)
								,Nodes->get_pos(n3,2));

			Tets->set_volume(t,tempVol);
		}
	
	
		//calculate effective volume of each node
		int i;
		for(int t = 0; t < Ntets; t++)
		{
			tempVol = 0.25 * Tets->get_volume(t);
			for (int tn = 0; tn < 4; tn++)
			{
				i = Tets->get_nab(t,tn);
				Nodes->add_volume(i,tempVol);
			}
		}

		//normalize volume so that each node
		//has an average volume of 1
		//i_Node.normalize_volume(float(Nnodes));

		//calculate total volume
		Tets->calc_total_volume();
		
		return true;
	}
	catch (const std::exception& e)
	{
		// print something
		return false;
	}
}

bool Mesh::CalculateAinv()
{
	try
	{
		init_As(*Nodes, *Tets);
		return true;
	}
	catch (const std::exception& e)
	{
		// print something
		return false;
	}
}
