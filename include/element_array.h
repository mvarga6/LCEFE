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
    virtual void assert_idx_access(int idx); // makes sure we're not adding beyond size
    void reset_element(int idx); // Resets an element to a default init state

    static const size_t Ntags = 2;

public:
    int **Elements;     // size x DoF
    int **Tags;         // group tags on the element
    size_t size;        // number of elements
    ElementType type;   // type of the elements in this array

    ///
    /// Requires an outer size and type to create
    ///

    MeshElementArray(const int N, ElementType type);
    ~MeshElementArray();

    ///
    /// Allocated the element data dimension and assigns values to it
    ///

    // just with element node indices
    void push_back(int node_idx[DoF]);

    // with element node indices and group tags
    void push_back(int node_idx[DoF], int ntags, int *tags);

    // remove the last element
    void pull_back();

    // returns pointer to nodes of element idx
    const int* element_at(int idx);

    // return tags for element idx
    const int* tags_for(int idx);

    // swaps two elements
    virtual void swap(int A_idx, int B_idx);
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

    // allocate outer
    this->Elements = new int*[size];
    this->Tags     = new int*[size];
    for (int i = 0; i < size; i++)
    {
        // allocate DoF inner elements
        this->Elements[i] = new int[DoF];
        this->Tags[i] = new int[Ntags]; // assume two tags per item
        reset_element(i);
    }    
}


template<size_t DoF>
MeshElementArray<DoF>::~MeshElementArray()
{
    // clear inner
    for (int i = 0; i < size; i++)
    {
        delete[] this->Elements[i];
        delete[] this->Tags[i];
    }

    // clear outer
    delete[] this->Elements;
    delete[] this->Tags;
}


template<size_t DoF>
void MeshElementArray<DoF>::push_back(int node_idx[DoF])
{
    assert_idx_access(add_idx);
    
    // assign elements
    for (int j = 0; j < DoF; j++)
    {
        this->Elements[add_idx][j] = node_idx[j];
    }
    add_idx++; // increment for next add
}


template<size_t DoF>
void MeshElementArray<DoF>::push_back(int node_idx[DoF], int ntags, int *tags)
{
    assert_idx_access(add_idx);

    // assign elements
    for (int j = 0; j < DoF; j++)
    {
        this->Elements[add_idx][j] = node_idx[j];
    }

    // assign tags
    for (int j = 0; j < std::min((size_t)ntags, Ntags); j++)
    {
        this->Tags[add_idx][j] = tags[j];
    }

    add_idx++; // increment for next add
}


template<size_t DoF>
void MeshElementArray<DoF>::pull_back()
{
    add_idx--; // ptr to last element
    reset_element(add_idx); // reset it
}


template<size_t DoF>
const int* MeshElementArray<DoF>::element_at(int idx)
{
    assert_idx_access(idx);
    return this->Elements[idx];
}


template<size_t DoF>
const int* MeshElementArray<DoF>::tags_for(int idx)
{
    assert_idx_access(idx);
    return this->Tags[idx];
}


template<size_t DoF>
void MeshElementArray<DoF>::swap(int A_idx, int B_idx)
{
    assert_idx_access(A_idx);
    assert_idx_access(B_idx);

    // swap by swapping pointer values
    int* _tmp = this->Elements[A_idx];
    this->Elements[A_idx] = this->Elements[B_idx];
    this->Elements[B_idx] = _tmp;

    _tmp = this->Tags[A_idx];
    this->Tags[A_idx] = this->Tags[B_idx];
    this->Tags[B_idx] = _tmp;
}


///
/// PROTECTED METHODS
///

template<size_t DoF>
void MeshElementArray<DoF>::reset_element(int idx)
{
    assert_idx_access(idx);

    // -1 is signature of unitizalized
    for (int j = 0; j < DoF; j++)
    {
        this->Elements[idx][j] = -1;
    }

    // Max # of tags is set by static const int
    for (int j = 0; j < Ntags; j++)
    {
        this->Tags[idx][j] = -1;
    }
}


template<size_t DoF>
void MeshElementArray<DoF>::assert_idx_access(int idx)
{
    if (idx >= this->size)
        throw std::runtime_error("MeshElementArray size is too small to add another element.");
}

#endif