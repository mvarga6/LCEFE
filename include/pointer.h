#ifndef __POINTER_H__
#define __POINTER_H__

///
/// Enum for location of data of PointerHandle
///
enum class MemoryLocation : int
{
    UNALLOCATED = 0,
    HOST = 1,
    DEVICE = 2
};

///
/// Class to wrapper a data point to carry along with
/// it some metadata, like where it's allocated (gpu, cpu)
///
template<typename T>
class PointerHandle
{
protected:
    MemoryLocation location; /// memory location
    T *ptr;                  /// the data pointer
    int size;                /// the number of elements

public:

    ///
    /// Required default constructor
    PointerHandle(){};

    ///
    /// Copy constructor
    PointerHandle(const PointerHandle&);

    ///
    /// Useful constructor to set private variables
    PointerHandle(T *ptr, int size, MemoryLocation location);

    ///
    /// Assignment operator overload
    PointerHandle<T>& operator=(PointerHandle<T>& rhs);

    ///
    /// Get the memory location of the data
    MemoryLocation Location();

    ///
    /// Get the actual data pointer
    T* Ptr();

    ///
    /// Get the number of data elements
    size_t Size();
};

///
/// Pointer handle to CPU allocated data pointer
///
template<typename T>
class CpuPointerHandle : public PointerHandle<T>
{
public:
    CpuPointerHandle(){};
    CpuPointerHandle(T *ptr, int size);
    CpuPointerHandle(const CpuPointerHandle<T> &);
};

///
/// Pointer handle to GPU allocated data pointer
///
template<typename T>
class GpuPointerHandle : public PointerHandle<T>
{
public:
    GpuPointerHandle(){};
    GpuPointerHandle(T *ptr, int size);
    GpuPointerHandle(const GpuPointerHandle<T> &);
};

template <typename T>
PointerHandle<T>::PointerHandle(const PointerHandle<T> &copy)
{
    this->ptr = copy.ptr;
    this->size = copy.size;
    this->location = copy.location;
}

template<typename T>
PointerHandle<T>::PointerHandle(T *ptr, int size, MemoryLocation location)
{
    this->ptr = ptr;
    this->location = location;
    this->size = size;
}

template<typename T>
PointerHandle<T>& PointerHandle<T>::operator=(PointerHandle<T> &rhs)
{
    this->ptr = rhs.ptr;
    this->location = rhs.location;
    this->size = rhs.size;
    return (*this);
}

template<typename T>
MemoryLocation PointerHandle<T>::Location()
{
    return this->location;
}

template<typename T>
T* PointerHandle<T>::Ptr()
{
    return this->ptr;
}

template<typename T>
size_t PointerHandle<T>::Size()
{
    return (size_t)this->size;
}

template<typename T>
CpuPointerHandle<T>::CpuPointerHandle(T *ptr, int size) 
    : PointerHandle<T>(ptr, size, MemoryLocation::HOST){}


template<typename T>
CpuPointerHandle<T>::CpuPointerHandle(const CpuPointerHandle<T> &copy)
{
    this->ptr = copy.ptr;
    this->size = copy.size;
    this->location = copy.location;
}


template<typename T>
GpuPointerHandle<T>::GpuPointerHandle(T *ptr, int size) 
    : PointerHandle<T>(ptr, size, MemoryLocation::DEVICE){}

template<typename T>
GpuPointerHandle<T>::GpuPointerHandle(const GpuPointerHandle<T> &copy)
{
    this->ptr = copy.ptr;
    this->size = copy.size;
    this->location = copy.location;
}

#endif