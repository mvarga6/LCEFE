#ifndef __MESH_OPTIMIZER_H__
#define __MESH_OPTIMIZER_H__

#include "classstruct.h"
#include "logger.h"
#include "director_field.h"
#include "defines.h"

enum class OperationResult : int
{
	SUCCESS = 1,
	FAILURE_NO_OPTIMATION = 2,
	FAILURE_EXCEPTION_THROWN = 3
};

/**
 * Abstract parent of all optimization operations
 * to run on the mesh
 */
class MeshOperation
{
public:
	virtual OperationResult Run(TetArray*, NodeArray*, Logger*) = 0;
};


/**
 * Optimize the mesh by sorting it based on
 * positions of tetrahedra.
 */
class SortOnTetrahedraPosition : public MeshOperation
{
public:
	OperationResult Run(TetArray*, NodeArray*, Logger*);
};

/**
 * Minimize distances between tets and their neighbors
 * by using simulated annealing.
 */
class MonteCarloMinimizeDistanceBetweenPairs : public MeshOperation
{
	real kbt_start, kbt_end, anneal_factor;
public:
	MonteCarloMinimizeDistanceBetweenPairs(const real kBTStart, const real kBTEnd, const real annealFactor);
	OperationResult Run(TetArray*, NodeArray*, Logger*);
};

/**
 * Reassign all the node and tet numbers in their
 * current configuration in memory.
 */
class ReassignIndices : public MeshOperation
{
public:
	OperationResult Run(TetArray*, NodeArray*, Logger*);
};

/**
 * Set the director in the mesh using an operation
 */
class SetDirector : public MeshOperation
{
	DirectorField *director;
public:
	SetDirector(DirectorField*);
	OperationResult Run(TetArray*, NodeArray*, Logger*);
};

/**
 * Calculate the Shape functions for the mesh in an operation.
 */
class CalculateAinv : public MeshOperation
{
public:
	OperationResult Run(TetArray*, NodeArray*, Logger*);
};

/**
 * Calculate the tet volumes in the mesh using an operation.
 */
class CalculateVolumes : public MeshOperation
{
public:
	OperationResult Run(TetArray*, NodeArray*, Logger*);
};

class EulerRotation : public MeshOperation
{
	real theta, phi, rho;
public:
	EulerRotation(const real, const real, const real);
	OperationResult Run(TetArray*, NodeArray*, Logger*);
};


#endif
