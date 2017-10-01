#ifndef __MESH_H__
#define __MESH_H__

#include "classstruct.h"
#include "simulation_parameters.h"
#include "mesh_optimizer.h"
#include "director_field.h"
#include <string>

using namespace std;



class Mesh
{
	bool loaded;

public:

	// temperarily public
	TetArray *Tets;
	NodeArray *Nodes;
	SimulationParameters *params; // a ptr to the simulation parameters

	Mesh(SimulationParameters *);
	
	bool Load();
	void SetDirector(DirectorField *);
	void Apply(MeshOptimizer *);
	void Update();
};

#endif