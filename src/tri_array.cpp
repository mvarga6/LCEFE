#include "tri_array.h"
#include <stdexcept>

TriArray::TriArray(const int N)
{
    if (N < 1) throw std::runtime_error("Creating TriArray requires a position, non-zero size.");

    this->size = N;
    this->NodeIdx = new int*[size];
    this->Com = new real*[size];
    this->Normal = new real*[size];
    this->Area = new real[size];
    this->TotalArea = 0;

    for (int i = 0; i < size; i++)
    {
        this->NodeIdx[i] = new int[3];
        this->Com[i] = new real[3];
        this->Normal[i] = new real[3];
        this->Area[i] = (real)0;
        for(int j = 0; j < 3; j++)
        {
            this->NodeIdx[i][j] = -1;
            this->Com[i][j] = (real)0;
            this->Normal[i][j] = (real)0;
        }
    }
}


TriArray::~TriArray()
{
    // delete the 2D array
    for (int i = 0; i < size; i++)
    {
        delete[] this->NodeIdx[i];
        delete[] this->Com[i];
        delete[] this->Normal[i];
    }

    delete[] this->NodeIdx;
    delete[] this->Com;
    delete[] this->Normal;
    delete[] this->Area;
}


void TriArray::set_nodes(int tri_idx, int n1_idx, int n2_idx, int n3_idx)
{
    assert_idx_access(tri_idx);
    this->NodeIdx[tri_idx][0] = n1_idx;
    this->NodeIdx[tri_idx][1] = n2_idx;
    this->NodeIdx[tri_idx][2] = n3_idx;
}


void TriArray::set_node_idx(int tri_idx, int n_i, int node_idx)
{
    assert_property_access(tri_idx, n_i);
    this->NodeIdx[tri_idx][n_i] = node_idx;
}


int& TriArray::node_idx(int tri_idx, int n_i)
{
    assert_property_access(tri_idx, n_i);
    return this->NodeIdx[tri_idx][n_i];
}


void TriArray::set_com(int tri_idx, real comx, real comy, real comz)
{
    assert_idx_access(tri_idx);
    this->Com[tri_idx][0] = comx;
    this->Com[tri_idx][1] = comy;
    this->Com[tri_idx][2] = comz;
}


void TriArray::set_com(int tri_idx, int dim, real com_d)
{
    assert_property_access(tri_idx, dim);
    this->Com[tri_idx][dim] = com_d;
}


real& TriArray::com(int tri_idx, int dim)
{
    assert_property_access(tri_idx, dim);
    return this->Com[tri_idx][dim];
}


void TriArray::set_normal(int tri_idx, real N_x, real N_y, real N_z)
{
    assert_idx_access(tri_idx);
    this->Normal[tri_idx][0] = N_x;
    this->Normal[tri_idx][1] = N_y;
    this->Normal[tri_idx][2] = N_z;
}


void TriArray::set_normal(int tri_idx, int dim, real N_d)
{
    assert_property_access(tri_idx, dim);
    this->Normal[tri_idx][dim] = N_d;
}


real& TriArray::normal(int tri_idx, int dim)
{
    assert_property_access(tri_idx, dim);
    return this->Normal[tri_idx][dim];
}


void TriArray::set_area(int tri_idx, real area)
{
    assert_idx_access(tri_idx);
    this->Area[tri_idx] = area;
}


real& TriArray::area(int tri_idx)
{
    assert_idx_access(tri_idx);
    return this->Area[tri_idx];
}


real TriArray::dist(int triA_idx, int triB_idx)
{
    assert_idx_access(triA_idx);
    assert_idx_access(triB_idx);
    real dx = this->Com[triA_idx][0] - this->Com[triB_idx][0];
    real dy = this->Com[triA_idx][1] - this->Com[triB_idx][1]; 
    real dz = this->Com[triA_idx][2] - this->Com[triB_idx][2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}


void TriArray::swap(int triA_idx, int triB_idx)
{
    assert_idx_access(triA_idx);
    assert_idx_access(triB_idx);
    int buf_idx;
    real buf_com, buf_area, buf_n;
    for (int d = 0; d < 3; d++)
    {
        // swap the d-th node idx of two triangles
        buf_idx = this->NodeIdx[triA_idx][d];
        this->NodeIdx[triA_idx][d] = this->NodeIdx[triB_idx][d];
        this->NodeIdx[triB_idx][d] = buf_idx;

        // swap the d-th com position of two triangles
        buf_com = this->Com[triA_idx][d];
        this->Com[triA_idx][d] = this->Com[triB_idx][d];
        this->Com[triB_idx][d] = buf_com;

        // swap the d-th normal component of two triangles
        buf_n = this->Normal[triA_idx][d];
        this->Normal[triA_idx][d] = this->Normal[triB_idx][d];
        this->Normal[triB_idx][d] = buf_n;
    }

    // swap the areas of two triangles
    buf_area = this->Area[triA_idx];
    this->Area[triA_idx] = this->Area[triB_idx];
    this->Area[triB_idx] = buf_area;
}


void TriArray::reorder(std::vector<int> const &order)
{   
    int _size = order.size();
    if (_size != this->size)
        throw std::runtime_error("The size of the new proposed order does not match the current size.");

    for ( int s = 0, d; s < _size; ++s) 
    {
        for ( d = order[s]; d < s; d = order[d]);
        if (d == s) 
        {
        	while (d = order[d], d != s ) 
        	{
        		this->swap(s, d);
        	}	
        }
    }
}


real TriArray::update_total_area(real *nodePositions)
{
    throw std::runtime_error("TriArray::update_total_area() not implemented!");
    return 0;
}


real TriArray::update_total_area(NodeArray *nodes)
{
    throw std::runtime_error("TriArray::update_total_area() not implemented!");
    return 0;
}


void TriArray::update_normals(real *nodePositions)
{
    throw std::runtime_error("TriArray::update_normals() not implemented!");
}


void TriArray::update_normals(NodeArray *nodes)
{
    throw std::runtime_error("TriArray::update_normals() not implemented!");
}


void TriArray::update_coms(real *nodePositions)
{
    throw std::runtime_error("TriArray::update_coms() not implemented!");
}


void TriArray::update_coms(NodeArray *nodes)
{
    throw std::runtime_error("TriArray::update_coms() not implemented!");
}




///
/// PRIVATE METHODS
///


void TriArray::assert_idx_access(int idx)
{
    if (idx < 0 || idx >= size)
        throw std::runtime_error("idx is out-of-range accessing TriArray element.");
}

void TriArray::assert_dim_access(int dim)
{
    if (dim < 0 || dim > 2)
        throw std::runtime_error("dim is out-of_range accessing TriArray property.");
}

void TriArray::assert_property_access(int idx, int dim)
{
    assert_idx_access(idx);
    assert_dim_access(dim);
}
