#ifndef __TRI_ARRAY_H__
#define __TRI_ARRAY_H__

#include "defines.h"
#include "node_array.h"

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

template<size_t DoF>
class MeshElementArray
{
public:
    int **Elements;     // size x DoF
    int **Tags;         // group tags on the element
    real **R;           // a 3d position for each element
    size_t size;        // number of elements
    ElementType type;   // type of the elements

    ///
    /// Setters/Getters for elements
    ///

    virtual void set_element(int idx, int dim, const int val) = 0;
    virtual int get_element(int idx, int dim) = 0;
    virtual int& element(int idx, int dim) = 0;

    ///
    /// Setters/Getters for element group tags
    ///

    virtual void set_tag(int idx, int tid, const int tag) = 0;
    virtual int get_tag(int idx, int tid) = 0;
    virtual int& tag(int idx, int tid) = 0;

    ///
    /// Setters/Getters for elment position (usually C.o.m)
    ///

    virtual void set_pos(int idx, int dim, const real& val) = 0;
    virtual void set_pos(int idx, real vals[DoF]) = 0;
    virtual real get_pos(int idx, int dim) = 0;
    virtual real& pos(int idx, int dim) = 0;
};

class TriArray
{

public:
    int **NodeIdx;
    real **Com;
    real **Normal;
    real *Area;
    size_t size;
    real TotalArea;

    TriArray(const int N);
    ~TriArray();

    ///
    /// Setters/Getters for node indices
    ///

    void set_nodes(int tri_idx, int n1_idx, int n2_idx, int n3_idx);
    void set_node_idx(int tri_idx, int n_i, int node_idx);
    int& node_idx(int tri_idx, int n_i);

    ///
    /// Setters/Getters for triangle positions
    ///

    void set_com(int tri_idx, real comx, real comy, real comz);
    void set_com(int tri_idx, int dim, real com_d);
    real& com(int tri_idx, int dim);

    ///
    /// Setters/Getters for triangle normals
    ///

    void set_normal(int tri_idx, real N_x, real N_y, real N_z);
    void set_normal(int tri_idx, int dim, real N_d);
    real& normal(int tri_idx, int dim);

    ///
    /// Setters/Getters for triangle areas
    ///

    void set_area(int tri_idx, real area);
    real& area(int tri_idx);

    ///
    /// Helper methods
    ///

    real dist(int triA_idx, int triB_idx);
    void swap(int triA_idx, int triB_idx);
    void reorder(std::vector<int> const &order);

    ///
    /// Methods for internal updates using the passed
    /// node positions for calculates
    ///

    real update_total_area(real *nodePositions);
    real update_total_area(NodeArray *nodes);
    void update_normals(real *nodePositions);
    void update_normals(NodeArray *nodes);
    void update_coms(real *nodePositions);
    void update_coms(NodeArray *nodes);

private:

    ///
    /// Methods that throw exceptions of the passed
    /// arguments will cause out of bounds errors
    ///

    void assert_idx_access(int idx);
    void assert_dim_access(int dim);
    void assert_property_access(int idx, int dim);
};

#endif