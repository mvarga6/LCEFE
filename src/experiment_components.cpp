#include "experiment.h"
#include <exception>
#include "errorhandle.h"
#include "defines.h"
#include "math_kernels.h"

#include "cuda.h"
#include "cuda_runtime.h"

///
/// Experiment Component Implementations
///

NematicToIsotropic::NematicToIsotropic(
    real tStart, 
    real tStop,
    PointerHandle<real> S)
{
    this->t_start = tStart;
    this->t_stop = tStop;
    this->s = S;

    // calculate the rate at which S drops to isotropic
    this->dsdt = -1 / (tStop - tStart);
}

bool NematicToIsotropic::Update(real dt)
{
    bool success = false;

    switch(this->s.Location())
    {
        case MemoryLocation::HOST:
            success = this->UpdateCpu(dt);
            break;

        case MemoryLocation::DEVICE:
            success = this->UpdateGpu(dt);
            break;

        case MemoryLocation::UNALLOCATED:
            throw std::runtime_error("Cannot operate on PointerHandle that is not bound to allocated memory.");

        default:
            success = false;
            break;
    }

    // add to aggregate time
    this->t += dt;

    return success;
}

bool NematicToIsotropic::UpdateCpu(real dt)
{
    // get the pointer
    real *_s = this->s.Ptr();
    int size = this->s.Size();
    real dS = this->dsdt * dt;
    for (int i = 0; i < size; i++)
    {
        _s[i] += dS;
    }
    return true;
}

bool NematicToIsotropic::UpdateGpu(real dt)
{
    real *_s = this->s.Ptr();
    int size = this->s.Size();
    real dS = this->dsdt * dt;

    int tpb = DefaultThreadsPerBlock;
    int blocks = (size / tpb) + 1;

    Add<<<blocks, tpb>>>(_s, dS, size);

    HANDLE_ERROR(cudaGetLastError());

    return true;
}