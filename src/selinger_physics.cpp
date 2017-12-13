#include "physics_model.h"
#include "physics_kernels.h"
#include "errorhandle.h"

#include "cuda.h"

void SelingerPhysics::CalculateForces(DataManager *data, real time)
{
    auto dims = data->TetKernelDimensions();
    ForceKernel<<<dims.BlockArrangement, dims.ThreadArrangement>>>(
        *(data->DeviceData()), time
    );
    HANDLE_ERROR(cudaGetLastError());
}

void SelingerPhysics::UpdateSystem(DataManager *data)
{
    auto dims = data->NodeKernelDimensions();
    UpdateKernel<<<dims.BlockArrangement, dims.ThreadArrangement>>>(
        *(data->DeviceData())
    );
    HANDLE_ERROR(cudaGetLastError());
}