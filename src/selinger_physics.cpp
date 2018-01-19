#include "physics_model.h"
#include "physics_kernels.h"
#include "errorhandle.h"
#include "helpers_math.h"
//#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "cuda.h"

// struct apply_pressure_functor
// {
//     const real P;

//     apply_pressure_functor(real pressure) : P(pressure){}

//     //template <typename Tuple>
//     __host__ __device__
//     void operator()(thrust::tuple<real&, int&, real&, real&> t)
//     {
//         // zip 0 = Surface Trianlge Normal
//         // zip 1 = TriNodeRank
//         // zip 2 = TriArea
//         // zip 3 = dF (forces on nodes split up by rank)

//         const real N[3] = {thrust::get<0>(t), thrust::get<0>(t), thrust::get<0>(t)};

//         // D[i] = A[i] + B[i] * C[i];
//         //thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
        
        
//     }
// };


void SelingerPhysics::CalculateForces(DataManager *data, real time)
{
    ///
    /// Calculates Elastic and LC forces
    ///
    DevDataBlock * dev = data->DeviceData();

    auto tetdims = data->TetKernelDimensions();
    BulkForceKernel<<<tetdims.BlockArrangement, tetdims.ThreadArrangement>>>(
        *dev, time
    );
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    ///
    /// Calculate Pressure forces
    ///

    // calculate constributes to enclosed volume
    auto tridims = data->TriKernelDimensions();
    CalculateClosedVolumesKernel<<<tridims.BlockArrangement, tridims.ThreadArrangement>>>(
        *dev, make_float3((real)0.0, (real)0.0, (real)0.0)
    );
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    //
    // Reduce the enclosed volume array with thrust library
    //

    // bind to thrust pointers
    static thrust::device_ptr<real> dev_vol_ptr = thrust::device_pointer_cast(dev->EnclosedVolume);

    // reduce volume contributions to total enclosed volume
    const real V = thrust::reduce(dev_vol_ptr, dev_vol_ptr + dev->Ntris);
    cudaThreadSynchronize();

    // calculate pressure
    static const real k_pressure = (real)0.1;
    static const real V0 = dev->InitialEnclosedVolume;

    // Apply pressure forces based on calculated volume
    PressureForcesKernel<<<tridims.BlockArrangement,tridims.ThreadArrangement>>>(
       *dev, V, V0, k_pressure
    );

    cudaThreadSynchronize();
}

void SelingerPhysics::UpdateSystem(DataManager *data)
{
    auto dims = data->NodeKernelDimensions();
    UpdateKernel<<<dims.BlockArrangement, dims.ThreadArrangement>>>(
        *(data->DeviceData())
    );
    HANDLE_ERROR(cudaGetLastError());
}