#ifndef __MESH_OPTIMIZER_H__
#define __MESH_OPTIMIZER_H__

#include "classstruct.h"
#include "logger.h"

enum class OptimizationResult : int
{
	SUCCESS = 1,
	FAILURE_NO_OPTIMATION = 2,
	FAILURE_EXCEPTION_THROWN = 3
};

/*
  Abstract parent of all optimization operations
  to run on the mesh
*/
class MeshOptimizer
{
public:
	virtual OptimizationResult Run(TetArray*, NodeArray*) = 0;
};


/*
  Optimize the mesh by sorting it based on
  positions of tetrahedra.
*/
class SortOnTetrahedraPosition : public MeshOptimizer
{
	Logger *log;
public:
	SortOnTetrahedraPosition(Logger *log);
	OptimizationResult Run(TetArray*, NodeArray*);
};

/*
  Minimize distances between tets and their neighbors
  by using simulated annealing.
*/
class MonteCarloMinimizeDistanceBetweenPairs : public MeshOptimizer
{
	float kbt_start, kbt_end, anneal_factor;
	Logger *log;
public:
	MonteCarloMinimizeDistanceBetweenPairs(const float kBTStart, const float kBTEnd, const float annealFactor, Logger *log);
	OptimizationResult Run(TetArray*, NodeArray*);
};

/*
  Reassign all the node and tet numbers in their
  current configuration in memory.
*/
class ReassignIndices : public MeshOptimizer
{
	Logger *log;
public:
	ReassignIndices(Logger *log);
	OptimizationResult Run(TetArray*, NodeArray*);
};

#endif
