#include "tri_array.h"
#include <stdexcept>
#include "helpers_math.h"

TriArray::TriArray(const int N) : MeshElementArray(N, TRIANGLE)
{
    if (N < 1) throw std::runtime_error("Creating TriArray requires a position, non-zero size.");

    //this->size = N;
    //this->NodeIdx = new int*[size];
    this->NodeIdx = this->Elements; // a TriArray proxy for elments array
    this->NodeRank = new int*[size];
    this->Com = new real*[size];
    this->Normal = new real*[size];
    this->NormalSign = new int[size];
    this->Area = new real[size];
    this->TotalArea = 0;

    for (int i = 0; i < size; i++)
    {
        //this->NodeIdx[i] = new int[3];
        this->NodeRank[i] = new int[3];
        this->Com[i] = new real[3];
        this->Normal[i] = new real[3];
        this->NormalSign[i] = 1;
        this->Area[i] = (real)0;
        for(int j = 0; j < 3; j++)
        {
            //this->NodeIdx[i][j] = -1;
            this->NodeRank[i][j] = 0;
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
        //delete[] this->NodeIdx[i];
        delete[] this->Com[i];
        delete[] this->Normal[i];
    }

    //delete[] this->NodeIdx;
    delete[] this->Com;
    delete[] this->Normal;
    delete[] this->Area;
}


TriArray* TriArray::SelectTag(int tag)
{
    const int _size = this->size;

    ///
    /// Calculate how to make new TriArray
    ///

    // loop all triangles
    std::vector<int> idx_to_keep;
    for (int idx = 0; idx < _size; idx++)
    {
        // loop triangle's tags
        for (int t = 0; t < Ntags; t++)
        {
            // grab if it match tag
            int element_tag = Tags[idx][t];
            if (tag == element_tag)
            {
                idx_to_keep.push_back(idx);
            }
        }
    }

    // size of the new tri array
    const int new_size = idx_to_keep.size();

    // stop if we didn't fine anything
    if (new_size == 0) 
        return NULL;

    // alloc new
    TriArray * result = new TriArray(new_size);

    // copy values into new array
    int this_idx;
    for (int idx = 0; idx < new_size; idx++)
    {
        // get the idx of the old element
        this_idx = idx_to_keep.at(idx);

        // copy values into new object
        result->Area[idx] = this->Area[this_idx];

        // vector properties
        for (int i = 0; i < 3; i++)
        {
            result->NodeIdx[idx][i] = this->NodeIdx[this_idx][i];
            result->NodeRank[idx][i] = this->NodeRank[this_idx][i];
            result->Com[idx][i] = this->Com[this_idx][i];
            result->Normal[idx][i] = this->Normal[this_idx][i];
            if (i < Ntags)
            {
                result->Tags[idx][i] = this->Tags[this_idx][i];
            }
        }
    }
    return result;
}


void TriArray::set_nodes(int idx, int n1_idx, int n2_idx, int n3_idx, int tag)
{
    assert_idx_access(idx);
    this->NodeIdx[idx][0] = n1_idx;
    this->NodeIdx[idx][1] = n2_idx;
    this->NodeIdx[idx][2] = n3_idx;
    this->Tags[idx][0] = tag;
}


void TriArray::set_node_idx(int idx, int n_i, int node_idx)
{
    assert_property_access(idx, n_i);
    this->NodeIdx[idx][n_i] = node_idx;
}

void TriArray::set_tag(int idx, int tag)
{
    assert_idx_access(idx);
    this->Tags[idx][0] = tag;
}


int& TriArray::node_idx(int idx, int n_i)
{
    assert_property_access(idx, n_i);
    return this->NodeIdx[idx][n_i];
}


void TriArray::set_rank(int idx, int n_i, int rank)
{
    assert_property_access(idx, n_i);
    this->NodeRank[idx][n_i] = rank;
}

int& TriArray::rank(int idx, int n_i)
{
    assert_property_access(idx, n_i);
    return this->NodeRank[idx][n_i];
}


void TriArray::set_com(int idx, real comx, real comy, real comz)
{
    assert_idx_access(idx);
    this->Com[idx][0] = comx;
    this->Com[idx][1] = comy;
    this->Com[idx][2] = comz;
}


void TriArray::set_com(int idx, int dim, real com_d)
{
    assert_property_access(idx, dim);
    this->Com[idx][dim] = com_d;
}


real& TriArray::com(int idx, int dim)
{
    assert_property_access(idx, dim);
    return this->Com[idx][dim];
}


void TriArray::set_normal(int idx, real N_x, real N_y, real N_z)
{
    assert_idx_access(idx);
    
    // normalize
    real Nmag = sqrt(N_x*N_x + N_y*N_y + N_z*N_z);

    this->Normal[idx][0] = N_x / Nmag;
    this->Normal[idx][1] = N_y / Nmag;
    this->Normal[idx][2] = N_z / Nmag;
}


void TriArray::set_normal(int idx, int dim, real N_d)
{
    assert_property_access(idx, dim);
    this->Normal[idx][dim] = N_d;
}


real& TriArray::normal(int idx, int dim)
{
    assert_property_access(idx, dim);
    return this->Normal[idx][dim];
}

int TriArray::normal_sign(int idx, float3 ref)
{
    // get position of triangle
    return this->NormalSign[idx];
}


void TriArray::set_area(int idx, real area)
{
    assert_idx_access(idx);
    this->Area[idx] = area;
}


real& TriArray::area(int idx)
{
    assert_idx_access(idx);
    return this->Area[idx];
}


real TriArray::dist(int A_idx, int B_idx)
{
    assert_idx_access(A_idx);
    assert_idx_access(B_idx);
    real dx = this->Com[A_idx][0] - this->Com[B_idx][0];
    real dy = this->Com[A_idx][1] - this->Com[B_idx][1]; 
    real dz = this->Com[A_idx][2] - this->Com[B_idx][2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}


void TriArray::swap(int A_idx, int B_idx)
{
    assert_idx_access(A_idx);
    assert_idx_access(B_idx);

    // do parent swap
    MeshElementArray::swap(A_idx, B_idx);

    //int buf_idx;
    //real buf_com, buf_n;

    // // additional swapping for triangles
    // for (int d = 0; d < 3; d++)
    // {
    //     // swap the d-th node idx of two triangles
    //     buf_idx = this->NodeIdx[A_idx][d];
    //     this->NodeIdx[A_idx][d] = this->NodeIdx[B_idx][d];
    //     this->NodeIdx[B_idx][d] = buf_idx;
    //
    //     // swap the d-th com position of two triangles
    //     buf_com = this->Com[A_idx][d];
    //     this->Com[A_idx][d] = this->Com[B_idx][d];
    //     this->Com[B_idx][d] = buf_com;
    //
    //     // swap the d-th normal component of two triangles
    //     buf_n = this->Normal[A_idx][d];
    //     this->Normal[A_idx][d] = this->Normal[B_idx][d];
    //     this->Normal[B_idx][d] = buf_n;
    // }

    // fast swap (swap pointers rather than values)
    real *tmp_real;

    // swap c.o.m
    tmp_real = this->Com[A_idx];
    this->Com[A_idx] = this->Com[B_idx];
    this->Com[B_idx] = tmp_real;

    // swap normals
    tmp_real = this->Normal[A_idx];
    this->Normal[A_idx] = this->Normal[B_idx];
    this->Normal[B_idx] = tmp_real;

    // swap the areas of two triangles
    real buf_area = this->Area[A_idx];
    this->Area[A_idx] = this->Area[B_idx];
    this->Area[B_idx] = buf_area;
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


real TriArray::update_areas(real *nodePositions)
{
    //throw std::runtime_error("TriArray::update_total_area() not implemented!");
    return 0;
}


real TriArray::update_areas(NodeArray *nodes)
{
    this->TotalArea = 0;
    real P, A, r12, r13, r23;
    int n1, n2, n3;
    for (int i = 0; i < size; i++)
    {
        // get node indices
        n1 = this->Elements[i][0];
        n2 = this->Elements[i][1];
        n3 = this->Elements[i][2];

        // calculate distances
        r12 = nodes->dist(n1, n2);
        r13 = nodes->dist(n1, n3);
        r23 = nodes->dist(n2, n3);

        P = (r12 + r23 + r13) / (real)2.0;
        A = (real)sqrt(P * (P - r12) * (P - r23) * (P -r13));
        this->set_area(i, A);
        this->TotalArea += A;
    }
    return this->TotalArea;
}


void TriArray::update_normals(real *nodePositions)
{
    throw std::runtime_error("TriArray::update_normals() not implemented!");
}


void TriArray::update_normals(NodeArray *nodes, float3 ref)
{
    real N[3], r12[3], r13[3];
    int n1, n2, n3;
    for (int i = 0; i < size; i++)
    {
        // get node indices
        n1 = this->Elements[i][0];
        n2 = this->Elements[i][1];
        n3 = this->Elements[i][2];

        // calculate displacements
        nodes->disp(n1, n2, r12);
        nodes->disp(n1, n3, r13);

        // cross product
        N[0] = r12[1] * r13[2] - r12[2] * r13[1];
        N[1] = r12[2] * r13[0] - r12[0] * r13[2];
        N[2] = r12[0] * r13[1] - r12[1] * r13[0];

        this->set_normal(i, N[0], N[1], N[2]);

        // centor of mass of the triangle
        float3 CoM = nodes->centroid({n1, n2, n3});

        // position relative to reference
        float3 Rref = ref - CoM;
        real _dot = dot(Rref, make_float3(N[0], N[1], N[2]));

        // is normal pointing parallel or antiparallel to R
        if (_dot < 0)
        {
            this->NormalSign[i] = 1;
        }
        else this->NormalSign[i] = -1;
    }
}


void TriArray::update_coms(real *nodePositions)
{
    throw std::runtime_error("TriArray::update_coms() not implemented!");
}


void TriArray::update_coms(NodeArray *nodes)
{
    real CoM[3];
    int n1, n2, n3;
    for (int i = 0; i < size; i++)
    {
        // get node indices
        n1 = this->Elements[i][0];
        n2 = this->Elements[i][1];
        n3 = this->Elements[i][2];

        for (int d = 0; d < 3; d++)
        {
            // average of this dimension
            CoM[d] = (nodes->get_pos(n1, d)
                    + nodes->get_pos(n2, d)
                    + nodes->get_pos(n3, d)) 
                    / (real)3;
        }

        this->set_com(i, CoM[0], CoM[1], CoM[2]);
    }
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
