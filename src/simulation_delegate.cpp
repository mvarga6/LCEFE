#include "simulation_delegate.h"

#include "functions.hpp"
#include "constant_cuda_defs.h"
#include "mesh_optimizer.h"

// these will go away into their own service class
#include "getAs.h"
#include "printmeshorder.h"
#include "packdata.h"
#include "errorhandle.h"
#include "datatodevice.h"
#include "anyerrors.h"
#include "exit_program.h"

SimulationDelegate::SimulationDelegate()
{
	// alloc private variables
	parameters = new SimulationParameters();	
	performance = new PerformanceRecorder();
	
	// data on host and dev
	dev = new DevDataBlock();
	host = new HostDataBlock();
	
	// inject dev and host datablocks into the data manager
	this->dataManager = new DataManager(this->host, this->dev);
}

ProcessInputResult SimulationDelegate::ProcessUserInput(int argc, char *argv[])
{
	performance->Create("UserInput")->Start();

	// Get parameters from command line
	ParametersReader * reader = new ParametersReader();
	
	// read cmdline	for parameters file	
	ParseResult result = reader->ReadFromCmdline(argc, argv, *parameters);
	if (result != ParseResult::SUCCESS)
	{
		Logger::Default->Msg(ParametersReader::GetErrorMsg(result));
		return ProcessInputResult::INVALID_INPUT;
	}
	
	// from file if given
	if (!parameters->File.empty())
	{
		result = reader->ReadFromFile(parameters->File, *parameters);
		if (result != ParseResult::SUCCESS)
		{
			Logger::Default->Msg(ParametersReader::GetErrorMsg(result));
			return ProcessInputResult::INVALID_INPUT;
		}
	}
	
	// override parameter files values with cmdline values
	result = reader->ReadFromCmdline(argc, argv, *parameters);
	if (result != ParseResult::SUCCESS)
	{
		Logger::Default->Msg(ParametersReader::GetErrorMsg(result));
		return ProcessInputResult::INVALID_INPUT;
	}
	
	
	
	//
	// Process user input for director field
	//
	
	UnivariableFunction * phi_of_z = new Linear({PI / 6}, PI / 4);
	ScalerField3D 		* theta	   = new ConstantField3D(PI / 4);
	ScalerField3D		* phi 	   = new AdditiveField3D(NULL, NULL, phi_of_z);
	
	this->director = new CartesianDirectorField(theta, phi);
	
	// Stop performance tracking on this step
	performance->Stop("UserInput");	
	
	return ProcessInputResult::TEST;
}


InitializeResult SimulationDelegate::Initialize()
{
	performance->Create("Initialize")->Start();
	
	// switch on loggertype from usering input
	switch(parameters->Output.LogType)
	{
	case LoggerType::NONE:
		printf("Logging disabled");
		break;
		
	case LoggerType::CONSOLE:
		logger = new ConsoleLogger(parameters->Output.LogLevel);
		break;
		
	case LoggerType::FILE:
		logger = new ConsoleLogger(parameters->Output.LogLevel);
		logger->Msg("File logger not implemented! Converting to console logger.", LogEntryPriority::WARNING);
		break;
	}	
	
	// create the vtkWriter
	vtkWriter = new VtkWriter(parameters->Output.Base);

	// create the mesh
	this->mesh = new Mesh(parameters, logger);
	
	// load the mesh
	bool cachedMesh;
	if (!mesh->Load(&cachedMesh))
	{
		// log failure
		exit(10);
	}
	
	// we haven't cached an optimized version of the mesh
	if (!cachedMesh)
	{
		// optimize the mesh
		MeshOptimizer * simpleSort = new SortOnTetrahedraPosition();
		MeshOptimizer * mcReorder = new MonteCarloMinimizeDistanceBetweenPairs(300, 0.01f, 0.99999f);
		MeshOptimizer * reIndex = new ReassignIndices();
		this->mesh->Apply(simpleSort);
		this->mesh->Apply(mcReorder);
		this->mesh->Apply(reIndex);
		
		// save the optimized mesh
		this->mesh->Cache();
	}
	
	// ste the director in the mesh
	this->mesh->SetDirector(this->director);
	
	// calculate volumes in the mesh
	if (!(this->mesh->CalculateVolumes()))
	{
		return InitializeResult::FAILURE;
	}
	
	// calculate inverse of tetrahedra shapes
	if (!(this->mesh->CalculateAinv()))
	{
		return InitializeResult::FAILURE;
	}
	
	// print the director
	this->mesh->Tets->printDirector(this->parameters->Output.Base);
	
	// move mesh data to host and dev blocks
	packdata(*mesh->Nodes, *mesh->Tets, host, &surfaceTetraIds, parameters);
	data_to_device(dev, host, parameters, dataManager);	
	
	logger->Msg("Simulation initialized");
	performance->Stop("Initialize");	
	
	return InitializeResult::SUCCESS;
}


int SimulationDelegate::Run()
{
	int Threads_Per_Block = parameters->Gpu.ThreadsPerBlock;
	int Blocks = (mesh->Tets->size + Threads_Per_Block) / Threads_Per_Block;
	int *Syncin, *Syncout, *g_mutex, *SyncZeros;
	
	HANDLE_ERROR( cudaMalloc( (void**)&Syncin, Blocks*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&Syncout, Blocks*sizeof(int) ) );
	SyncZeros = (int*)malloc(Blocks*sizeof(int));
	
	for (int i = 0; i < Blocks; i++)
	{
		SyncZeros[i]=0;
	}
	
	HANDLE_ERROR( cudaMemcpy(Syncin, SyncZeros, Blocks*sizeof(int), cudaMemcpyHostToDevice ) );
	//allocate global mutex and set =0 
	HANDLE_ERROR( cudaMalloc( (void**)&g_mutex, sizeof(int) ) );
	HANDLE_ERROR( cudaMemset( g_mutex, 0, sizeof(int) ) );

	// run the simulation
	run_dynamics(dev, 
		host,
		parameters, 
		Syncin, 
		Syncout, 
		g_mutex, 
		&surfaceTetraIds, 
		vtkWriter, 
		dataManager, 
		performance);
		
	any_errors();
	
	HANDLE_ERROR(cudaFree( Syncin ) );
	HANDLE_ERROR(cudaFree( Syncout ) );
	HANDLE_ERROR(cudaFree( g_mutex ) );
	exit_program(dev);

	return 1;
}
