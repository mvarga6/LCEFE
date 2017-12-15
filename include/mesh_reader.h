#ifndef __MESH_READER_H__
#define __MESH_READER_H__

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cmath>
#include <cstring>
#include <string>
#include "extlib/gmsh_io/gmsh_io.hpp"
#include "classstruct.h"
#include "tri_array.h"
#include "parameters.h"
#include "logger.h"

enum ElementType : int
{
	LINE = 1,
	TRIANGLE = 2,
	QUADRANGLE = 3,
	TETRAHEDRON = 4,
	HEXAHEDRON = 5,
	PRISM = 6,
	PYRAMID = 7,
	POINT = 15
};

struct MeshDimensions
{
	int Ntets, Nnodes, Ntris;
	real rmin[3], rmax[3];
};

class MeshReader
{
public:
    MeshReader(Logger *log);
};

#endif