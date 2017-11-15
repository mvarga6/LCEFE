#include "mesh.h"
#include "getgmsh.h"
#include "getmesh.h"
#include "getAs.h"

#include "file_operations.hpp"

Mesh::Mesh(SimulationParameters *parameters, Logger *log)
{
	this->loaded = false;
	this->params = parameters;
	this->log = log;
}


bool Mesh::Load(bool *loadedFromCache)
{
	// try to read from cached
	if (this->params->Mesh.CachingOn)
	{
		std::string key = GetCacheKey();
		if (ReadCache(key))
		{
			(*loadedFromCache) = true;
			return true;
		}
	}

	// read the mesh given in parameters
	if (LoadMesh(this->params->Mesh.File))
	{
		(*loadedFromCache) = false;
		return true;
	}
	
	// could not read it from cache or 
	// from parameters given mesh file
	return false;
}


bool Mesh::Cache()
{
	std::string key = GetCacheKey();
	return WriteCache(key);
}


void Mesh::SetDirector(DirectorField *field)
{
	int Ntets = this->Tets->size;
	
	DirectorOrientation dir;
	real x, y, z;
	
	for (int t = 0; t < Ntets; t++)
	{
		// the position of this tet
		x = this->Tets->get_pos(t, 0);
		y = this->Tets->get_pos(t, 1);
		z = this->Tets->get_pos(t, 2);
	
		// get the director there
		// done this way so it could be read
		// from file or hardcoded etc
		dir = field->GetDirectorAt(x, y, z);
		
		// assign 
		this->Tets->set_theta(t, dir.theta);
		this->Tets->set_phi(t, dir.phi);
	}
}



void Mesh::Apply(MeshOperation *operation)
{
	OperationResult result;
	try
	{
		result = operation->Run(this->Tets, this->Nodes, this->log);
		switch(result)
		{
		case OperationResult::SUCCESS:
			// log success
			return;
		
		case OperationResult::FAILURE_NO_OPTIMATION:
			// log no optimization
			return;
		
		case OperationResult::FAILURE_EXCEPTION_THROWN:
			// log exception
			exit(101);
		
		default:
			// log unknown result
			return;
		}
	}
	catch (const std::exception& e)
	{
		this->log->Error(e.what());
		return;
	}
}



bool Mesh::CalculateVolumes()
{
	try
	{
		real tempVol;
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
		//i_Node.normalize_volume(real(Nnodes));

		//calculate total volume
		Tets->calc_total_volume();
		
		return true;
	}
	catch (const std::exception& e)
	{
		this->log->Error(e.what());
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
		this->log->Error(e.what());
		return false;
	}
}


bool Mesh::LoadMesh(const std::string &meshFile)
{
	stringstream ss;
	ss << "Loading mesh " << meshFile;
	this->log->Msg(ss.str());

	//get dimensions of the mesh
	dimensions = new MeshDimensions;
	try
	{
		(*dimensions) = get_gmsh_dim(meshFile);
		
	}
	catch (const std::exception& e)
	{
		// print something
		this->log->Error(e.what());
		return false;
	}
	
	// unload if loaded
	if (loaded)
	{
		delete this->Tets;
		delete this->Nodes;
	}
	
	// allocate the containers for nodes and tets
	this->Tets = new TetArray(dimensions->Ntets);
	this->Nodes = new NodeArray(dimensions->Nnodes);	
	
	// read the positions of nodes and tet indices
	try
	{
		(*dimensions) = get_gmsh(meshFile, *Nodes, *Tets, this->params->Mesh.Scale);
	}
	catch (const std::exception& e)
	{
		this->log->Error(e.what());
		return false;
	}
	
	// read the positions of nodes and tet indices
	try
	{
		get_tet_pos(*Nodes, *Tets);
	}
	catch (const std::exception& e)
	{
		this->log->Error(e.what());
		return false;
	}
	
	loaded = true;
	return true;
}


bool Mesh::ReadCache(const std::string &cachedMeshFile)
{
	if (!FileOperations::Exists(cachedMeshFile))
	{
		// theres no cache for this mesh file
		stringstream ss;
		ss << "Cached mesh " << cachedMeshFile << " does not exist";
		this->log->Msg(ss.str());
		return false;
	}
	
	// load it like a normal mesh file
	return LoadMesh(cachedMeshFile);
}


bool Mesh::WriteCache(const std::string &cacheFileName)
{
	stringstream ss;

	if (FileOperations::Exists(cacheFileName))
	{
		// already cached
		ss << "Cached mesh " << cacheFileName << " already exists, abort overwriting it.";
		this->log->Msg(ss.str());
		return true;
	}
	
	try
	{
		ss << "Caching mesh " << cacheFileName;
		this->log->Msg(ss.str());
		int Nnodes = Nodes->size;
		int Ntets = Tets->size;
		
		// need to make copy of tet nab array 
		// gmsh_writer changes it to base-1
		int * element_node = new int[Ntets*4];
		for(int t = 0; t < Ntets; t++)
		{
			for(int n = 0; n < 4; n++)
			{
				element_node[n + t*4] = Tets->get_nab(t, n);
			}
		}
		
		// write the cached mesh file
		gmsh_mesh3d_write(cacheFileName, 3, Nnodes, (float*)Nodes->MyPos, 4, Ntets, element_node);
		
		delete [] element_node;
	}
	catch (const std::exception& e)
	{
		this->log->Error(e.what());
		return false;
	}
	
	return true;
}


std::string Mesh::GetCacheKey()
{
	std::string meshFile = this->params->Mesh.File;
	FileInfo info = FileOperations::GetFileInfo(meshFile);
	
	string key = "";
	if (!info.Path.empty())
	{
		key += info.Path + "/";
	}
	
	key += info.FileName + ".cache";
	return key;
}
