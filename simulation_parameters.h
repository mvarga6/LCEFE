#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

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

class SimulationParameters
{
public:
	
	std::string File;

	struct MaterialConstants
	{
		float Cxxxx;
		float Cxxyy;
		float Cxyxy;
		float Alpha;
		float Density;
	} Material;

	struct DynamicsParameters
	{
		int Nsteps;
		float Dt;
		float Damp;
	} Dynamics;

	struct GpuParameters
	{
		int ThreadsPerBlock;
	} Gpu;

	struct OutputParameters
	{
		std::string Base;
		int FrameRate;
	} Output;
	
	struct MeshParameters
	{
		std::string File;
		int NodeRankMax;
		float Scale;
	} Mesh;
	
	// parameters for the initial
	// setup of the LCE mesh
	struct InitialState
	{
		bool PlanarSideUp;
		float SqueezeAmplitude;
		float SqueezeRatio;
	} Initalize;
	
	struct ActuationParameters
	{
	 	// Order parameter dynamics
	 	struct OrderParameter
		{
			float SInital;
			float Smax;
			float Smin;
			float SRateOn;
			float SRateOff;
		} OrderParameter;
	
		// Parameters of UV illumination
		struct UVIllumination
		{
			float IncidentAngle;
			int IterPerIllumRecalc;
		} Optics;
			
	} Actuation;
};

#endif
