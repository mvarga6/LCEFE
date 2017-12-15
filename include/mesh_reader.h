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
#include "mesh.h"



struct MeshSpecs
{
    int Ntets;
    int Nnodes;
    int Ntris;
    real rmin[3];
    real rmax[3];
};

class MeshReader
{
public:
    MeshReader(Logger *log);
    virtual MeshSpecs ReadSpecs(string fileName) = 0;
    virtual bool Read(Mesh *mesh, MeshSpecs *specs = NULL) = 0;
    virtual bool Read(NodeArray *nodes, int Nnodes) = 0;
    virtual bool Read(TetArray *tets, int Ntets) = 0;
    virtual bool Read(TriArray *tris, int Ntris) = 0;
};

// class GmshReader : class MeshReader
// {
// public:
//     GmshReader(Logger * log) : MeshReader(log);
//     MeshSpecs ReadSpecs(string fileName)
// };

#endif