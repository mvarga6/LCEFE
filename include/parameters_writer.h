#ifndef __PARAMETERS_WRITER_H__
#define __PARAMETERS_WRITER_H__

#include <string>
#include "logger.h"
#include "simulation_parameters.h"

using namespace std;

class ParametersWriter
{
public:
	virtual bool Write(SimulationParameters*) = 0;
};

class JsonWriter : public ParametersWriter
{
	string file;
public:

	JsonWriter(const string&);
	bool Write(SimulationParameters*);
};

class ConsoleWriter : public ParametersWriter
{
	string file;
public:
	bool Write(SimulationParameters*);
};

class LogWriter : public ParametersWriter
{
	Logger * log;
public:
	LogWriter(Logger* logger);
	bool Write(SimulationParameters*);
};

#endif
