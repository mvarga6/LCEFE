
#include <iostream>
#include <fstream>

#include "parameters_writer.h"


using namespace std;

JsonWriter::JsonWriter(const string& fileName)
{
	this->file = fileName;
}

bool JsonWriter::Write(SimulationParameters& parameters)
{
	//
	//Implement writing to file
	//

	return true;
}

bool ConsoleWriter::Write(SimulationParameters& parameters)
{
	cout << endl;
	cout << "*************************" << endl;
	cout << "  Simulation Parameters  " << endl;
	cout << "*************************" << endl;
	cout << "File: " << parameters.File << endl << endl;
	
	cout << "o Dynamics" << endl;
	cout << "\tnsteps: " << parameters.Dynamics.Nsteps << endl;
	cout << "\tdt:     " << parameters.Dynamics.Dt << endl;
	cout << "\tdamp:   " << parameters.Dynamics.Damp << endl << endl;
	
	cout << "o Material" << endl;
	cout << "\tcxxxx:   " << parameters.Material.Cxxxx << endl;
	cout << "\tcxxyy:   " << parameters.Material.Cxxyy << endl;
	cout << "\tcxyxy:   " << parameters.Material.Cxyxy << endl;
	cout << "\talpha:   " << parameters.Material.Alpha << endl;
	cout << "\tdensity: " << parameters.Material.Density << endl << endl;
	
	cout << "o Gpu" << endl;
	cout << "\tthreadsferblock: " << parameters.Gpu.ThreadsPerBlock << endl << endl;
	
	cout << "o Output" << endl;
	cout << "\toutputbase: " << parameters.Output.Base << endl;
	cout << "\tframerate:  " << parameters.Output.FrameRate << endl << endl;
	
	cout << "o Mesh" << endl;
	cout << "\tmeshfile:  " << parameters.Mesh.File << endl;
	cout << "\tmeshscale: " << parameters.Mesh.Scale << endl << endl;
	
	return true;
}
