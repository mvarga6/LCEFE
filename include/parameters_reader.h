#ifndef __PARAMETERS_READER_H__
#define __PARAMETERS_READER_H__

#include <string>
#include "simulation_parameters.h"
#include <map>
// updated parameteres_reader.h
using namespace std;

enum ParseResult : int
{
	SUCCESS = 0,
	CRITICAL_FAILURE = 1,
	MESHFILE_NAME_IS_NULL = 2,
	ARGS_COUNT_ZERO = 3,
	ARGS_MISSING = 4
};

class ParametersReader
{
	ParseResult status;
	typedef map<ParameterType, string> tokenMap;
	typedef map<ParameterType, string>::iterator tokenMapIterator;

public:
	ParseResult ReadFromFile(const string&, SimulationParameters&);
	ParseResult ReadFromCmdline(int, char* [],SimulationParameters&);

private:

	tokenMap ParseCmdlineToTokenMap(int, char* []);
	tokenMap ParseFileToTokenMap(const string&);
	void ConvertTokenMapToParameters(tokenMap&, SimulationParameters&);
	 
	
	bool HandleParseResult(int);
	string JsmnTokenType(int);
	ParameterType GetParameterType(string&, bool&);
	bool CleanKey(string&);
	bool StrToBool(const string&);
};

#endif