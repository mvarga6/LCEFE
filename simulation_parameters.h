#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include <string>
#include "defines.h"

enum ParameterType : int
{
	Unknown = 0,
	Cxxxx = 1,
	Cxxyy = 2,
	Cxyxy = 3,
	Alpha = 4,
	Density = 5,
	Nsteps = 6,
	Dt = 7,
	Damp = 8,
	ThreadsPerBlock = 9, 
	OutputBase = 10,
	FrameRate = 11,
	MeshFile = 12,
	NodeRankMax = 13,
	MeshScale = 14,
	PlanarSideUp = 15,
	Amplitude = 16,
	Ratio = 17,
	HomeoSideUp = 18,
	SInitial = 19,
	Smax = 20,
	Smin = 21,
	SRateOn = 22,
	SRateOff = 23,
	IncidentAngle = 24,
	IterPerIllumRecalc = 25,
	ParametersFile = 26,
	MeshCaching = 27
};

struct MaterialConstants
{
	__host__
	MaterialConstants() {};
	
	real Cxxxx;
	real Cxxyy;
	real Cxyxy;
	real Alpha;
	real Density;
};

struct DynamicsParameters
{
	__host__
	DynamicsParameters(){};

	int Nsteps;
	real Dt;
	real Damp;
};

struct GpuParameters
{
	__host__
	GpuParameters() {};
	
	int ThreadsPerBlock;
};

struct OutputParameters
{
	__host__
	OutputParameters() {};	
	
	std::string Base;
	int FrameRate;
};

struct MeshParameters
{
	__host__
	MeshParameters() {};	
	
	std::string File;
	int NodeRankMax;
	real Scale;
	bool CachingOn;
};

struct InitialState
{
	__host__
	InitialState() {};

	bool PlanarSideUp;
	real SqueezeAmplitude;
	real SqueezeRatio;
};

struct LiquidCrystalParameters
{
	__host__
	LiquidCrystalParameters() {};
	
	real SInitial;
	real Smax;
	real Smin;
	real SRateOn;
	real SRateOff;
};

struct UVIllumination
{
	__host__
	UVIllumination() {};

	real IncidentAngle;
	int IterPerIllumRecalc;
};

struct ActuationParameters
{
	__host__
	ActuationParameters() {};

 	// Order parameter dynamics
 	LiquidCrystalParameters OrderParameter;

	// Parameters of UV illumination
	UVIllumination Optics;
};

struct SimulationParameters
{
	__host__
	SimulationParameters() {};

	// Name of the file with parameters 
	std::string File;
	
	// all constants for LCE material
	MaterialConstants Material;
	
	// parameters for running dynamics
	DynamicsParameters Dynamics;
	
	// parameters for GPU operation
	GpuParameters Gpu;
	
	// parameters for simulation output
	OutputParameters Output;
	
	// parmeters for the mesh
	MeshParameters Mesh;
	
	// parameters for initialization
	InitialState Initalize;
	
	// parameters LCE actuation
	ActuationParameters Actuation;
};

struct PackedParameters
{
	real Alpha;
	real Cxxxx, Cxxyy, Cxyxy;
	real Density;
	real Dt, Damp;
	real Scale;
	real SInitial, Smin, Smax, SRateOn, SRateOff;
	real IncidentAngle;
};

#endif
