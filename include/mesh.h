#ifndef __MESH_H__
#define __MESH_H__

#include "classstruct.h"
#include "tri_array.h"
#include "simulation_parameters.h"
#include "mesh_operations.h"
#include "director_field.h"
#include "logger.h"
#include <string>

using namespace std;

struct MeshDimensions;

class Mesh
{
	bool loaded;
	Logger *log;

public:

	// temperarily public
	TetArray *Tets;
	NodeArray *Nodes;
	TriArray *Tris;
	SimulationParameters *params; // a ptr to the simulation parameters
	MeshDimensions *dimensions;

	Mesh(SimulationParameters *, Logger *log);
	
	bool Load(bool*);
	bool Cache();
	void Apply(MeshOperation*);
	
private:
	
	bool LoadMesh(const std::string&);
	bool ReadCache(const std::string&);
	bool WriteCache(const std::string&);
	
	std::string GetCacheKey();
};

#endif
