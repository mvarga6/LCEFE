#include "data_procedures.h"

GetPrintData::GetPrintData()
{
	Operations.push_back(new PullPositionFromGpu());
	Operations.push_back(new PullVelocityFromGpu());
	Operations.push_back(new PullEnergyFromGpu());
}


PushAllToGpu::PushAllToGpu()
{
	Operations.push_back(new PushTetNodeRankToGpu());
	Operations.push_back(new PushThetaPhiToGpu());
	Operations.push_back(new PushNodeRankToGpu());
	Operations.push_back(new PushMassToGpu());
	Operations.push_back(new PushTetVolumeToGpu());
	Operations.push_back(new PushNematicOrderParameterToGpu());
	Operations.push_back(new PushAinvToGpu());
	Operations.push_back(new PushTetToNodeMapToGpu());
	Operations.push_back(new PushPostionToGpu());
	Operations.push_back(new PushReferencePositionToGpu());
	Operations.push_back(new PushVelocityToGpu());
	Operations.push_back(new PushForceToGpu());
	Operations.push_back(new BindReferencePositionTexture());
	Operations.push_back(new BindPositionTexture());
}
