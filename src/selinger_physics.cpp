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
//
    //// bind to thrust pointers
    //static thrust::device_ptr<real> dev_vol_ptr = thrust::device_pointer_cast(dev->EnclosedVolume);
    //static thrust::device_ptr<real> dev_area_ptr = thrust::device_pointer_cast(dev->TriArea);
    //
    //// reduce volume contributions
    //const real V = thrust::reduce(dev_vol_ptr, dev_vol_ptr + dev->Ntris);
//
    //// calculate pressure
    //static const real k_pressure = (real)0.1;
    //static const real V0 = dev->TargetEnclosedVolume;
    //real P = k_pressure * (V - V0) * (V - V0);
//
    //cudaThreadSynchronize();

    // add forces to nodes from pressure

    // block reduce to get total volume
    // static const int reduction_TPB = 128;
    // static int blocks = (dev->Ntris / reduction_TPB) + 1;

    // // allocate gpu memory for reduction stages
    // static bool reduction_memory_allocated = false;
    // static std::vector<real*> reduction_input;
    // static std::vector<real*> reduction_output;
    // static std::vector<int> reduction_N;
    // static int reduction_stages = 0;
    // static real * reduced_volume;
    // if (!reduction_memory_allocated)
    // {
    //     // first reduction is gaurenteed
    //     reduction_input.push_back(dev->EnclosedVolume);
    //     reduction_N.push_back(dev->Ntris);
    //     reduction_stages++;

    //     // alloc on gpu stage 1 output vector
    //     real *output = new real;
    //     HANDLE_ERROR(cudaMalloc((void**)&output, blocks*sizeof(real)));
    //     output->push_back(output); // store ptr to output memory

    //     // alloc the final reduced volume ptr
    //     HANDLE_ERROR(cudaMalloc((void**)&reduced_volume, sizeof(real)));

    //     // recalcuate blocks
    //     blocks = (blocks / reduction_TPB) + 1;

    //     // allocate the other stages of reduction
    //     while (blocks > 1)
    //     {
    //         // allocate another stage of reduction
    //         real *input = new real;
    //         reduction_input.push_back(input);
    //         reduction_N.push_back(blocks); // the current size of output



    //         // recalculate for next iteration
    //         blocks = (blocks / reduction_TPB) + 1;
    //         reduction_stages++;
    //     }
        
    //     reduction_memory_allocated = true;
    // }

    // // set the number of blocks at start of reductions
    // blocks = (dev->Ntris / reduction_TPB) + 1;
}

void SelingerPhysics::UpdateSystem(DataManager *data)
{
    auto dims = data->NodeKernelDimensions();
    UpdateKernel<<<dims.BlockArrangement, dims.ThreadArrangement>>>(
        *(data->DeviceData())
    );
    HANDLE_ERROR(cudaGetLastError());
}