#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include <string>
#include "defines.h"

///
/// Enum uniquely mapping to simulation parameters
///
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
	MeshCaching = 27,
	InitNoise = 28,
	StartExperiment = 29,
	StopExperiment = 30
};

///
/// Parameterse for material constants
///
struct MaterialConstants
{
	__host__
	MaterialConstants() {};
	
	real Cxxxx;		/// Principle elastic constant (On-diagonal elements) [ g / cm * s^2 ]
	real Cxxyy;		/// Elastic constant (first off-diagonal elements) [ g / cm * s^2 ]
	real Cxyxy;		/// Elastic constant (second off-diagonal elements) [ g / cm * s^2 ]
	real Alpha;		/// Coupling stength between LC director/order/strain [ g / cm * s^2 ]
	real Density;	/// Density of the material [ g / cm^3 ]

	static MaterialConstants Default()
	{
		MaterialConstants defaults;
		defaults.Cxxxx = (real)29400000;
		defaults.Cxxyy = (real)28000000;
		defaults.Cxyxy = (real)570000;
		defaults.Alpha = (real)1500000;
		defaults.Density = (real)1.2;
		return defaults;
	}
};

///
/// Parameters for the dynamics of the simulations
///
struct DynamicsParameters
{
	__host__
	DynamicsParameters(){};

	int Nsteps; /// Number of time steps total [ iterations ]
	int Start;  /// time step experiment start [ iterations ]
	int Stop; 	/// time step experiment ends [ iterations ]
	real Dt; 	/// Length of the time step [ s ]
	real Damp;	/// Velocity dampening scalor [ unitless ]

	__host__
	real ExperimentStart()
	{
		return Dt * real(Start);
	}

	real ExperimentStop()
	{
		return Dt * real(Stop);
	}

	static DynamicsParameters Default()
	{
		DynamicsParameters defaults;
		defaults.Nsteps = 10000;
		defaults.Start 	= 0;
		defaults.Stop 	= defaults.Nsteps;
		defaults.Dt 	= (real)0.00005;
		defaults.Damp 	= (real)0.999;
		return defaults;
	}
};

///
/// Parameters for gpu execution
///
struct GpuParameters
{
	__host__
	GpuParameters() {};
	
	int ThreadsPerBlock;	/// [64:512] the number of threads in a block [ thread count ]

	static GpuParameters Default()
	{
		GpuParameters defaults;
		defaults.ThreadsPerBlock = 256;
		return defaults;
	}
};

///
/// Parameteres for simulation output
///
struct OutputParameters
{
	__host__
	OutputParameters() {};	
	
	std::string Base;	/// Simulation output files all start with this base
	int FrameRate;		/// Simulation writes at this frequency [ iterations ]

	static OutputParameters Default()
	{
		OutputParameters defaults;
		defaults.Base 		= "sim";
		defaults.FrameRate 	= 1000;
		return defaults;
	}
};

///
/// Parameters for the mesh used in the simulation
///
struct MeshParameters
{
	__host__
	MeshParameters() {};	
	
	std::string File;	/// The name of the mesh file to load to run experiment on
	//int NodeRankMax;	/// [Obsolete] The maximum # of tets a node can belong to
	real Scale;			/// Scales the size of the mesh on load [ unitless ]
	bool CachingOn;		/// Sets if the mesh will be cached after optimization

	static MeshParameters Default()
	{
		MeshParameters defaults;
		defaults.File 		= "Mesh\\mesh.vtk";
		defaults.Scale 		= (real)1.0;
		defaults.CachingOn 	= true;
		return defaults;
	}
};

///
/// Parameters for the initialization of the simulation
///
struct InitialState
{
	__host__
	InitialState() {};

	real Noise;				/// [0:1] How much nodes are randomly moved after reading the mesh [ mesh unit ]
	bool PlanarSideUp;		/// From Broer, "Making waves with light" experiment
	real SqueezeAmplitude;	/// From Broer, "Making waves with light" experiment
	real SqueezeRatio;		/// From Broer, "Making waves with light" experiment

	static InitialState Default()
	{
		InitialState defaults;
		defaults.Noise 				= 0;
		defaults.PlanarSideUp 		= false;
		defaults.SqueezeAmplitude 	= 0;
		defaults.SqueezeRatio 		= 0;
		return defaults;
	}
};

///
/// Parameters for Liquid Crystal properties of the Mesh
///
struct LiquidCrystalParameters
{
	__host__
	LiquidCrystalParameters() {};
	
	real SInitial;	/// The initial order parameter assigned to each tet [ unitless ]
	real Smax;		/// The maximum allowed order parameter [ unitless ]
	real Smin;		/// the minimun allow order parameter [ unitless ]
	real SRateOn;	/// The rate at which S changes when illuminated. From Broer experiments. [ s^-1 ]
	real SRateOff;	/// The rate at which S changes when shadowed. From Broer experiments. [ s^-1 ]

	static LiquidCrystalParameters Default()
	{
		LiquidCrystalParameters defaults;
		defaults.SInitial 	= 0;
		defaults.Smax 		= (real)1.0;
		defaults.Smin 		= 0;
		defaults.SRateOn 	= (real)0.2;
		defaults.SRateOff 	= (real)-0.2;
		return defaults;
	}
};

///
/// Parameters for UV illumination on the sample
struct UVIllumination
{
	__host__
	UVIllumination() {};

	real IncidentAngle;		/// Incident angle of UV light. From Broer experiments [ deg ]
	int IterPerIllumRecalc;	/// Frequency of optics calculation [ iterations ]

	static UVIllumination Default()
	{
		UVIllumination defaults;
		defaults.IncidentAngle 		= 0;
		defaults.IterPerIllumRecalc = 1000;
		return defaults;
	}
};

///
/// Parameters defining how the LCE sample is actuated
///
struct ActuationParameters
{
	__host__
	ActuationParameters() {};

 	/// Order parameter dynamics
 	LiquidCrystalParameters OrderParameter;

	/// Parameters of UV illumination
	UVIllumination Optics;

	static ActuationParameters Default()
	{
		ActuationParameters defaults;
		defaults.OrderParameter = LiquidCrystalParameters::Default();
		defaults.Optics 		= UVIllumination::Default();
		return defaults;
	}
};

///
/// Groups all parameteres types into one object
///
struct SimulationParameters
{
	__host__
	SimulationParameters() {};

	__host__
	static SimulationParameters Default() 
	{
		SimulationParameters defaults;
		//defaults.File = "params.json";
		defaults.Material  = MaterialConstants::Default();
		defaults.Dynamics  = DynamicsParameters::Default();
		defaults.Gpu 	  = GpuParameters::Default();
		defaults.Output 	  = OutputParameters::Default();
		defaults.Mesh	  = MeshParameters::Default();
		defaults.Initalize = InitialState::Default();
		defaults.Actuation = ActuationParameters::Default();
		return defaults;
	};

	__host__
	static SimulationParameters* CreateDefault()
	{
		SimulationParameters * defaults = new SimulationParameters();
		//defaults->File = "params.json";
		defaults->Material  = MaterialConstants::Default();
		defaults->Dynamics  = DynamicsParameters::Default();
		defaults->Gpu 	  = GpuParameters::Default();
		defaults->Output 	  = OutputParameters::Default();
		defaults->Mesh	  = MeshParameters::Default();
		defaults->Initalize = InitialState::Default();
		defaults->Actuation = ActuationParameters::Default();
		return defaults;
	}

	/// Name of the file with parameters 
	std::string File;
	
	/// all constants for LCE material
	MaterialConstants Material;
	
	/// parameters for running dynamics
	DynamicsParameters Dynamics;
	
	/// parameters for GPU operation
	GpuParameters Gpu;
	
	/// parameters for simulation output
	OutputParameters Output;
	
	/// parmeters for the mesh
	MeshParameters Mesh;
	
	/// parameters for initialization
	InitialState Initalize;
	
	/// parameters LCE actuation
	ActuationParameters Actuation;
};

///
/// A flatten SimuationsParameteres object for use on the gpu
///
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
