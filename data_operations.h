#ifndef __DATA_OPERATIONS_H__
#define __DATA_OPERATIONS_H__

#include "datastruct.h"

/*
  Abstract parent of anything that moves data to/from gpu
*/
class DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*) = 0; };

/*
  DataOperation childen that act as they're names' suggest, PULL for gpu
*/
class PullPositionFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PullVelocityFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PullForceFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PullEnergyFromGpu : public DataOperation 
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

/*
  DataOperation childen that act as they're names' suggest, PUSH for gpu
*/
class PushTetNodeRankToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushThetaPhiToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushNodeRankToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushMassToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushTetVolumeToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushNematicOrderParameterToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushAinvToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushTetToNodeMapToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushPostionToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushReferencePositionToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushVelocityToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class PushForceToGpu : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };


/*
  Data Operations that Bind things to cuda textures
*/
class BindPositionTexture : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

class BindReferencePositionTexture : public DataOperation
{ public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

#endif
