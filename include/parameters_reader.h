#ifndef __PARAMETERS_READER_H__
#define __PARAMETERS_READER_H__

#include <string>
#include "simulation_parameters.h"
#include <map>

using namespace std;

enum ParseStatus : int
{
	SUCCESS = 0,
	CRITICAL_FAILURE = 1,
	PARAMETER_FILE_NAME_IS_EMTPY = 2,
	ARGS_COUNT_ZERO = 3,
	ARGS_MISSING = 4,
	PARAMETER_OBJ_NULL = 5,
	READY_TO_PARSE = 6
};

class ParseResult
{
public:
	string Message;
	ParseStatus Status;

	ParseResult()
	{
		Status = READY_TO_PARSE;
	}

	ParseResult(ParseStatus Status)
	{
		Status = Status;
	}

	ParseResult(ParseStatus Status, string message)
	{
		Status = Status;
		Message = message;
	}
};

class ParametersReader
{
	ParseResult result;

public:

	typedef map<ParameterType, string> tokenMap;
	typedef map<ParameterType, string>::iterator tokenMapIterator;

	SimulationParameters* ReadFromFile(const string& fileName);
	SimulationParameters* ReadFromCmdline(int argc, char* argv[]);
	SimulationParameters* Read(int argc, char* argv[]);
	
	ParseResult Result();
	ParseStatus Status();
	bool Success();

	ParseResult UpdateFromFile(const string& fileName, SimulationParameters* parameters);
	ParseResult UpdateFromCmdline(int argc, char* argv[], SimulationParameters* parameters);
	void UpdateFromTokenMap(tokenMap&, SimulationParameters*);

private:
	
	tokenMap ParseCmdlineToTokenMap(int, char* []);
	tokenMap ParseFileToTokenMap(const string&);
	 
	bool HandleParseResult(int, string&);
	string JsmnTokenType(int);
	ParameterType GetParameterType(string&, bool&);
	bool CleanKey(string&);
	bool StrToBool(const string&);
};

#endif
