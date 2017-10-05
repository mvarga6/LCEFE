#include "mesh.h"
#include "getgmsh.h"
#include "getmesh.h"
#include "getAs.h"
#include "logger.h"
#include "file_operations.hpp"
#include <sstream>

Mesh::Mesh(SimulationParameters *parameters, Logger * logger)
{
	this->loaded = false;
	this->params = parameters;
	this->logger = logger;
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
			logger->Log(new ErrorLog("Mesh loaded from cached version.", LogEntryPriority::INFO));
			return true;
		}
	}

	// read the mesh given in parameters
	if (LoadMesh(this->params->Mesh.File))
	{
		(*loadedFromCache) = false;
		logger->Log(new ErrorLog("Mesh loaded.", LogEntryPriority::INFO));
		return true;
	}
	
	// could not read it from cache or 
	// from parameters given mesh file
	logger->Log(new ErrorLog("Mesh NOT loaded.", LogEntryPriority::WARNING));
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
		logger->Log(new ErrorLog("Optimization failed.", e, LogEntryPriority::CRITICAL));
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
		logger->Log(new ErrorLog("Failed to calculate tetrahedra volumes.", e, LogEntryPriority::CRITICAL));
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
		logger->Log(new ErrorLog("Failed to invert tetrahedra matrices.", e, LogEntryPriority::CRITICAL));
		return false;
	}
}


bool Mesh::LoadMesh(const std::string &meshFile)
{
	stringstream ss;
	ss << "Loading mesh " << meshFile << "...";
	logger->Log(new ErrorLog(ss.str(), LogEntryPriority::DEBUG));

	//get dimensions of the mesh
	dimensions = new MeshDimensions;
	try
	{
		(*dimensions) = get_gmsh_dim(meshFile);
		
	}
	catch (const std::exception& e)
	{
		logger->Log(new ErrorLog("Failed to read mesh dimesnions.", e, LogEntryPriority::CRITICAL));
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
		logger->Log(new ErrorLog("Failed to read mesh positions.", e, LogEntryPriority::CRITICAL));
		return false;
	}
	
	// read the positions of nodes and tet indices
	try
	{
		get_tet_pos(*Nodes, *Tets);
	}
	catch (const std::exception& e)
	{
		logger->Log(new ErrorLog("Failed to calculate tetrahedra positions.", e, LogEntryPriority::CRITICAL));
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
		ss << "Cached mesh " << cachedMeshFile << " does not exist.";
		logger->Log(new ErrorLog(ss.str(), LogEntryPriority::DEBUG));
		return false;
	}
	
	// load it like a normal mesh file
	return LoadMesh(cachedMeshFile);
}


bool Mesh::WriteCache(const std::string &cacheFileName)
{
	if (FileOperations::Exists(cacheFileName))
	{
		// already cached
		stringstream ss;
		ss << "Cached mesh " << cacheFileName << " already exists.";
		logger->Log(new ErrorLog(ss.str(), LogEntryPriority::DEBUG));
		return true;
	}
	
	try
	{
		stringstream ss;
		ss << "Creating cache for mesh " << cacheFileName << ".";
		logger->Log(new ErrorLog(ss.str(), LogEntryPriority::INFO));
		
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
		gmsh_mesh3d_write(cacheFileName, 3, Nnodes, Nodes->MyPos, 4, Ntets, element_node);
		
		delete [] element_node;
	}
	catch (const std::exception& e)
	{
		logger->Log(new ErrorLog("Failed to write mesh cache.", e, LogEntryPriority::INFO));
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
