#ifndef __DATA_OPERATIONS_H__
#define __DATA_OPERATIONS_H__

#include "datastruct.h"

///
/// Abstract parent of anything that moves data to/from gpu
class DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*) = 0; };

///
/// Copies node position data on gpu to cpu constainers
class PullPositionFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node velocity data on gpu to cpu constainers 
class PullVelocityFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node force data on gpu to cpu constainers
class PullForceFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies tetrahedra energy data on gpu to cpu constainers
class PullEnergyFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node rank in tet container on cpu to gpu
class PushTetNodeRankToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies tetrahedra director data on cpu to gpu
class PushThetaPhiToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node rank data on cpu to gpu
class PushNodeRankToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node mass data on cpu to gpu
class PushMassToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies tet volume data on cpu to gpu
class PushTetVolumeToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies nematic order data on cpu to gpu
class PushNematicOrderParameterToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies tet shape function data on cpu to gpu
class PushAinvToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies tet-node map on cpu to gpu
class PushTetToNodeMapToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node position data on cpu to gpu
class PushPostionToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node reference position data on cpu to gpu
class PushReferencePositionToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node velocity data on cpu to gpu
class PushVelocityToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Copies node force data on cpu to gpu
class PushForceToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Binds position ptr to texture object on gpu
class BindPositionTexture : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

///
/// Binds reference position ptr to texture object on gpu
class BindReferencePositionTexture : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

#endif
