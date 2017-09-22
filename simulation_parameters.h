#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"

#include <string>

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
	ParametersFile = 26
};

struct MaterialConstants
{
	float Cxxxx;
	float Cxxyy;
	float Cxyxy;
	float Alpha;
	float Density;
};

struct DynamicsParameters
{
	int Nsteps;
	float Dt;
	float Damp;
};

struct GpuParameters
{
	int ThreadsPerBlock;
};

struct OutputParameters
{
	std::string Base;
	int FrameRate;
};

struct MeshParameters
{
	std::string File;
	int NodeRankMax;
	float Scale;
};

struct InitialState
{
	bool PlanarSideUp;
	float SqueezeAmplitude;
	float SqueezeRatio;
};

struct LiquidCrystalParameters
{
	float SInital;
	float Smax;
	float Smin;
	float SRateOn;
	float SRateOff;
};

struct UVIllumination
{
	float IncidentAngle;
	int IterPerIllumRecalc;
};

struct ActuationParameters
{
 	// Order parameter dynamics
 	LiquidCrystalParameters OrderParameter;

	// Parameters of UV illumination
	UVIllumination Optics;
};

struct SimulationParameters
{
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

#endif
