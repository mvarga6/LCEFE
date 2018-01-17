#ifndef __TRI_ARRAY_H__
#define __TRI_ARRAY_H__

#include "defines.h"
#include "node_array.h"
#include "element_array.h"

class TriArray : public MeshElementArray<3>
{

public:

    ///
    /// Each triangle storess idx of 3 nodes
    int **NodeIdx;

    ///
    /// The rank of the nodes in the triangle
    /// for writing forces to unique element 
    /// in force vectors.
    int **NodeRank;

    ///
    /// The center of mass of the triangle
    real **Com;

    ///
    /// The normal vector of the triangle
    real **Normal;

    ///
    /// The area of the triangle 
    real *Area;

    

    ///
    /// The total surface area of triangles
    real TotalArea;

    TriArray(const int N);
    ~TriArray();

    ///
    /// Creating new TriArrays
    ///

    TriArray* SelectTag(int tag);

    ///
    /// Setters/Getters for node indices
    ///

    void set_nodes(int idx, int n1_idx, int n2_idx, int n3_idx, int tag = 1);
    void set_node_idx(int idx, int n_i, int node_idx);
    int& node_idx(int idx, int n_i);

    ///
    /// Setters/Getters for the node ranks in these triangles
    ///

    void set_rank(int idx, int n_i, int rank);
    int& rank(int idx, int n_i);

    ///
    /// Setters/Getters for triangle positions
    ///

    void set_com(int idx, real comx, real comy, real comz);
    void set_com(int idx, int dim, real com_d);
    real& com(int idx, int dim);

    ///
    /// Setters/Getters for triangle normals
    ///

    void set_normal(int idx, real N_x, real N_y, real N_z);
    void set_normal(int idx, int dim, real N_d);
    real& normal(int idx, int dim);

    ///
    /// Setters/Getters for triangle areas
    ///

    void set_area(int idx, real area);
    real& area(int idx);

    ///
    /// Helper methods
    ///

    real dist(int A_idx, int B_idx);
    void swap(int A_idx, int B_idx);
    void reorder(std::vector<int> const &order);

    ///
    /// Methods for internal updates using the passed
    /// node positions for calculates
    ///

    real update_areas(real *nodePositions);
    real update_areas(NodeArray *nodes);
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