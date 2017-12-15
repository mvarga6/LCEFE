#ifndef __ELEMENT_ARRAY_H__
#define __ELEMENT_ARRAY_H__

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
protected:
    int add_idx;        // the idx of the next add
    void assert_add_access(int idx); // makes sure we're not adding beyond size

public:
    int **Elements;     // size x DoF
    int **Tags;         // group tags on the element
    size_t size;        // number of elements
    ElementType type;   // type of the elements in this array

    ///
    /// Requires an outer size and type to create
    ///

    MeshElementArray(const int N, ElementType type);

    ///
    /// Allocated the element data dimension and assigns values to it
    ///

    // just with element node indices
    virtual bool add_element(int node_idx[DoF]);

    // with element node indices and group tags
    bool add_element(int node_idx[DoF], int ntags, int *tags);
};

///
///
/// IMPLEMENATION
///
///


#include <stdexcept>


template<size_t DoF>
MeshElementArray<DoF>::MeshElementArray(const int N, ElementType type)
{
    this->size = N;
    this->type = type;
    this->add_idx = 0;

    // allocate outer dimensions only
    this->Elements = new int*[size];
    this->Tags     = new int*[size];
}


template<size_t DoF>
bool MeshElementArray<DoF>::add_element(int node_idx[DoF])
{
    assert_add_access(add_idx);
    
    // alloc and assign elements
    this->Elements[add_idx] = new int[DoF];
    for (int j = 0; j < DoF; j++)
    {
        this->Elements[add_idx][j] = node_idx[j];
    }
    add_idx++; // increment for next add
    return true;
}


template<size_t DoF>
bool MeshElementArray<DoF>::add_element(int node_idx[DoF], int ntags, int *tags)
{
    assert_add_access(add_idx);

    // alloc and assign elements
    this->Elements[add_idx] = new int[DoF];
    for (int j = 0; j < DoF; j++)
    {
        this->Elements[add_idx][j] = node_idx[j];
    }

    // alloc and assign tags
    this->Tags[add_idx] = new int[ntags];
    for (int j = 0; j < ntags; j++)
    {
        this->Tags[add_idx][j] = tags[j];
    }

    add_idx++; // increment for next add
    return true;
}


template<size_t DoF>
void MeshElementArray<DoF>::assert_add_access(int idx)
{
    if (idx >= this-> size)
        throw std::runtime_error("MeshElementArray size is too small to add another element.");
}

#endif