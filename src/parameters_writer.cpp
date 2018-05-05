
#include <iostream>
#include <fstream>

#include "parameters_writer.h"


using namespace std;

JsonWriter::JsonWriter(const string& fileName)
{
	this->file = fileName;
}

bool JsonWriter::Write(SimulationParameters* parameters)
{
	//
	//Implement writing to file
	//

	return true;
}

bool ConsoleWriter::Write(SimulationParameters* parameters)
{
	cout << endl;
	cout << "*************************" << endl;
	cout << "  Simulation Parameters  " << endl;
	cout << "*************************" << endl;
	cout << "File: " << parameters->File << endl << endl;
	
	cout << "o Dynamics" << endl;
	cout << "\tnsteps: " << parameters->Dynamics.Nsteps << endl;
	cout << "\tdt:     " << parameters->Dynamics.Dt << endl;
	cout << "\tdamp:   " << parameters->Dynamics.Damp << endl << endl;
	
	cout << "o Material" << endl;
	cout << "\tcxxxx:   " << parameters->Material.Cxxxx << endl;
	cout << "\tcxxyy:   " << parameters->Material.Cxxyy << endl;
	cout << "\tcxyxy:   " << parameters->Material.Cxyxy << endl;
	cout << "\talpha:   " << parameters->Material.Alpha << endl;
	cout << "\tdensity: " << parameters->Material.Density << endl << endl;
	
	cout << "o Gpu" << endl;
	cout << "\tthreadsferblock: " << parameters->Gpu.ThreadsPerBlock << endl << endl;
	
	cout << "o Output" << endl;
	cout << "\toutputbase: " << parameters->Output.Base << endl;
	cout << "\tframerate:  " << parameters->Output.FrameRate << endl << endl;
	
	cout << "o Mesh" << endl;
	cout << "\tmeshfile:  " << parameters->Mesh.File << endl;
	cout << "\tmeshscale: " << parameters->Mesh.Scale << endl;
	cout << "\tcaching: " << parameters->Mesh.CachingOn << endl << endl;
	
	return true;
}


LogWriter::LogWriter(Logger *logger)
{
	this->log = logger;
}


bool LogWriter::Write(SimulationParameters * parameters)
{
	log->Info("*************************");
	log->Info("  Simulation Parameters  ");
	log->Info("*************************");
	log->Info("[Parameter]       [Value]");
	log->Info("");
	log->Info(formatted("File:            %s", parameters->File.c_str()));
	log->Info("");
	log->Info("DYNAMICS");
	log->Info(formatted("nsteps:          %d", parameters->Dynamics.Nsteps));
	log->Info(formatted("dt:              %0.6f", parameters->Dynamics.Dt));
	log->Info(formatted("damp:            %0.6f", parameters->Dynamics.Damp));
	log->Info("");
	log->Info("MATERIAL");
	log->Info(formatted("cxxxx:           %0.0f", parameters->Material.Cxxxx));
	log->Info(formatted("cxxyy:           %0.0f", parameters->Material.Cxxyy));
	log->Info(formatted("cxyxy:           %0.0f", parameters->Material.Cxyxy));
	log->Info(formatted("alpha:           %0.0f", parameters->Material.Alpha));
	log->Info(formatted("density:         %0.2f", parameters->Material.Density));
	log->Info("");
	log->Info("GPU PARAMETERS");
	log->Info(formatted("threadsferblock: %d", parameters->Gpu.ThreadsPerBlock));
	log->Info("");
	log->Info("OUTPUT");
	log->Info(formatted("outputbase:      %s", parameters->Output.Base.c_str()));
	log->Info(formatted("framerate:       %d", parameters->Output.FrameRate));
	log->Info("");
	log->Info("MESH");
	log->Info(formatted("meshfile:        %s", parameters->Mesh.File.c_str()));
	log->Info(formatted("meshscale:       %0.2f", parameters->Mesh.Scale));
	log->Info(formatted("caching:         %s", (parameters->Mesh.CachingOn ? "ON" : "OFF")));
	log->Info("");
	return true;
}