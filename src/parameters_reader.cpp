
#include "parameters_reader.h"
#include "simulation_parameters.h"
#include "extlib/jsmn/jsmn.h"
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

ParametersReader::ParametersReader(Logger *logger)
{
	this->log = logger;
}

SimulationParameters* ParametersReader::ReadFromFile(const string& fileName)
{
	// create new parameters object
	SimulationParameters * parameters = SimulationParameters::CreateDefault();

	// update it from file
	UpdateFromFile(fileName, parameters);
	
	// confirm success or delete
	// if (!this->Success())
	// {
	// 	delete parameters;
	// 	return NULL;
	// }

	// return the ptr
	return parameters;
}

SimulationParameters* ParametersReader::ReadFromCmdline(int argc, char* argv[])
{
	// create new parameters object
	SimulationParameters * parameters = SimulationParameters::CreateDefault();

	// update it from cmdline
	UpdateFromCmdline(argc, argv, parameters);
	
	// confirm success or delete
	// if (!this->Success())
	// {
	// 	delete parameters;
	// 	return NULL;
	// }

	// return the ptr
	return parameters;
}

SimulationParameters* ParametersReader::Read(int argc, char* argv[])
{
	// create new parameters object
	SimulationParameters * parameters = SimulationParameters::CreateDefault();

	// update it from cmdline (so we can get a file to read from)
	UpdateFromCmdline(argc, argv, parameters);

	//confirm success or delete
	if (!this->Success())
	{
		delete parameters;
		return NULL;
	}

	// if we read a file name to read from
	if (!parameters->File.empty())
	{
		// update it from file
		UpdateFromFile(parameters->File, parameters);
		
		//confirm success or delete
		if (!this->Success())
		{
			delete parameters;
			return NULL;
		}

		// update it from cmdline again to override whats in file
		// shouldn't fail if it didn't already
		//UpdateFromCmdline(argc, argv, parameters);
	}

	// return the ptr
	return parameters;

}

ParseResult ParametersReader::Result()
{
	return this->result;
}

ParseStatus ParametersReader::Status()
{
	return this->result.Status;
}

bool ParametersReader::Success()
{
	return (this->result.Status == ParseStatus::SUCCESS) || (this->result.Status == ParseStatus::READY_TO_PARSE);
}

ParseResult ParametersReader::UpdateFromFile(const string& fileName, SimulationParameters* parameters)
{
	if (fileName.empty())
	{
		return (this->result = ParseResult(ParseStatus::PARAMETER_FILE_NAME_IS_EMTPY, "The passed file name is empty.  Please specify a file to read from using 'input' option."));
	}

	if (parameters == NULL)
	{
		return (this->result = ParseResult(ParseStatus::PARAMETER_OBJ_NULL, "The passed SimulationParameters ptr is null.  Please allocate a new object before trying to update it."));
	}

	tokenMap map = ParseFileToTokenMap(fileName);
	UpdateFromTokenMap(map, parameters);
	return this->result;
}

ParseResult ParametersReader::UpdateFromCmdline(int argc, char* argv[], SimulationParameters* parameters)
{
	if (argc < 1)
	{
		return (this->result = ParseResult(ParseStatus::ARGS_COUNT_ZERO, "There were no command line argument passed to the program. Please specify at least a parameters file name with 'input' option."));
	}
	
	if (argv == NULL)
	{
		return (this->result = ParseResult(ParseStatus::ARGS_MISSING, "The passed argument array is null.  Please confirm there were command line arguments passed to the program."));
	}

	if (parameters == NULL)
	{
		return (this->result = ParseResult(ParseStatus::PARAMETER_OBJ_NULL, "The passed SimulationParameters ptr is null.  Please allocate a new object before trying to update it."));
	}
	
	tokenMap map = ParseCmdlineToTokenMap(argc, argv);
	UpdateFromTokenMap(map, parameters);
	return this->result;
}


ParametersReader::tokenMap ParametersReader::ParseCmdlineToTokenMap(int argc, char* argv[])
{
	// the map we're constructing
	tokenMap pairs;

	// all cmdline arguments
	for (int i = 1; i < argc; i++) 
	{
		std::string arg = argv[i];
		std::string val;
		
		bool flag;
		ParameterType type = GetParameterType(arg, flag);
		if (type != Unknown)
		{
			// add to map
			if(flag)
			{
				pairs[type];
			}
			else
			{
				pairs[type] = string(argv[1 + i++]);
			}
		}
	}
	
	this->result = ParseResult(ParseStatus::SUCCESS);
	return pairs;
}


ParametersReader::tokenMap ParametersReader::ParseFileToTokenMap(const string& fileName)
{
	// the map we're constructing
	tokenMap pairs;
	
	// open the file
	ifstream fin(fileName.c_str());
	if (!fin.is_open())
	{
		this->result = ParseResult(ParseStatus::CRITICAL_FAILURE, "Could not open file '" + fileName + "'.  Confirm the path the parameters file.");
		return pairs;
	}
	
	// read json content into a string
	stringstream jsonStream;
	jsonStream << fin.rdbuf();
	string json = jsonStream.str();
	const char* js = json.c_str();
	//string json( istreambuf_iterator<char>(fin), istreambuf_iterator<char>());
	
	// initalize json parser
	jsmn_parser parser;
	jsmn_init(&parser);
	
	// read the number of tokens
	int num_tokens = jsmn_parse(&parser, js, strlen(js), NULL, 0);
	
	// check result
	string err_message;
	if (!HandleParseResult(num_tokens, err_message))
	{
		this->result = ParseResult(ParseStatus::CRITICAL_FAILURE, err_message);
		return pairs;
	}
	
	// create tokens array and reset parser
	jsmntok_t *tokens = new jsmntok_t[num_tokens];
	jsmn_init(&parser);
	
	// actually read the tokens now
	int result = jsmn_parse(&parser, js, strlen(js), tokens, num_tokens);
	
	// check result
	if (!HandleParseResult(result, err_message))
	{
		this->result = ParseResult(ParseStatus::CRITICAL_FAILURE, err_message);
		return pairs;
	}

	for (int t = 0; t < num_tokens; t++)
	{	
		// get the key and value
		jsmntok_t keyToken = tokens[t++];
		
		// only for value types we want
		if (keyToken.type == JSMN_STRING || keyToken.type == JSMN_PRIMITIVE)
		{
			// get the value now
			jsmntok_t valueToken = tokens[t];
			
			if (valueToken.type == JSMN_STRING || valueToken.type == JSMN_PRIMITIVE)
			{
				// get the actual key and value now
				string key(js + keyToken.start, js + keyToken.end);
				string value(js + valueToken.start, js + valueToken.end);
				
				// add to map for known types
				bool flag;
				ParameterType type = GetParameterType(key, flag);
				
				if (type == Unknown)
				{
					log->Warning(formatted("Parameter key (%s) not supported", key.c_str()));
					continue;
				}

				// add to map
				if (flag)
				{
					pairs[type];
				}
				else
				{
					pairs[type] = value;
				}
			}
		}
	}
	
	fin.close();
	
	this->result = ParseResult(ParseStatus::SUCCESS);
	return pairs;
}


void ParametersReader::UpdateFromTokenMap(tokenMap &map, SimulationParameters *p)
{
	tokenMapIterator begin = map.begin();
	tokenMapIterator end = map.end();
	tokenMapIterator it;
	
	for(it = begin; it != end; it++)
	{
		string v = it->second;
	
		switch(it->first)
		{
		case Unknown: 	break;
		case ParametersFile: p->File = v; break;
		case Cxxxx: 	p->Material.Cxxxx = ::atof(v.c_str()); break;
		case Cxxyy: 	p->Material.Cxxyy = ::atof(v.c_str()); break;
		case Cxyxy: 	p->Material.Cxyxy = ::atof(v.c_str()); break;
		case Alpha: 	p->Material.Alpha = ::atof(v.c_str()); break;
		case Density: 	p->Material.Density = ::atof(v.c_str()); break;
		case Nsteps:	p->Dynamics.Nsteps = ::atoi(v.c_str()); break;
		case Dt: 		p->Dynamics.Dt = ::atof(v.c_str()); break;
		case Damp: 		p->Dynamics.Damp = ::atof(v.c_str()); break;
		case ThreadsPerBlock:		p->Gpu.ThreadsPerBlock = ::atoi(v.c_str()); break;
		case OutputBase: 	p->Output.Base = v; break;
		case FrameRate: 	p->Output.FrameRate = ::atoi(v.c_str()); break;
		case MeshFile:		p->Mesh.File = v; break;
//		case NodeRankMax: 	p->Mesh.NodeRankMax = ::atoi(v.c_str()); break;
		case MeshScale: 	p->Mesh.Scale = ::atof(v.c_str()); break;
		case MeshCaching:	p->Mesh.CachingOn = StrToBool(v); break;
		case PlanarSideUp: 	p->Initalize.PlanarSideUp = true; break;
		case HomeoSideUp: 	p->Initalize.PlanarSideUp = false; break;
		case Amplitude: 	p->Initalize.SqueezeAmplitude = ::atof(v.c_str()); break;
		case Ratio: 		p->Initalize.SqueezeRatio = ::atof(v.c_str()); break;
		case SInitial: 		p->Actuation.OrderParameter.SInitial = ::atof(v.c_str()); break;
		case Smax: 			p->Actuation.OrderParameter.Smax = ::atof(v.c_str()); break;
		case Smin: 			p->Actuation.OrderParameter.Smin = ::atof(v.c_str()); break;
		case SRateOn: 		p->Actuation.OrderParameter.SRateOn = ::atof(v.c_str()); break;
		case SRateOff: 		p->Actuation.OrderParameter.SRateOff = ::atof(v.c_str()); break;
		case IncidentAngle: p->Actuation.Optics.IncidentAngle = ::atof(v.c_str()); break;
		case IterPerIllumRecalc: p->Actuation.Optics.IterPerIllumRecalc = ::atoi(v.c_str()); break;
		case InitNoise: 	p->Initalize.Noise = ::atof(v.c_str()); break;
		case StartExperiment: p->Dynamics.Start = ::atoi(v.c_str()); break;
		case StopExperiment: p->Dynamics.Stop = ::atoi(v.c_str()); break;
		default: break;
		}
	} 
}


bool ParametersReader::HandleParseResult(int result, string& err_msg)
{
	bool success = false;
	switch(result)
	{
	case JSMN_ERROR_NOMEM:
		err_msg = string("JSMN ERROR: Not enough tokens provided");
		break;
		
	case JSMN_ERROR_INVAL:
		err_msg = string("JSMN ERROR: Invalid character in parameters file");
		break;
		
	case JSMN_ERROR_PART:
		err_msg = string("JSMN ERROR: Incomplete JSON packet, expected more bytes");
		break;
		
	default:
		success = true;
		break;
	}
	if (!success) log->Error(err_msg);
	return success;
}

string ParametersReader::JsmnTokenType(int type)
{
	switch(type)
	{
	case JSMN_UNDEFINED:
		return string("undefined");
		
	case JSMN_OBJECT:
		return string("object");
		
	case JSMN_ARRAY:
		return string("array");
		
	case JSMN_STRING:
		return string("string");
		
	case JSMN_PRIMITIVE:
		return string("primitive");
		
	default:
		return string("error");		
	}
}

ParameterType ParametersReader::GetParameterType(string& key, bool &flagType)
{
	// set to true if param is a bool type
	flagType = false;

	// convert to lower
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (!CleanKey(key))
	{
		return Unknown;
	}
	
	if (key == "input" || key == "params" || key == "parameters") return ParametersFile;
	else if (key == "alpha" || key == "alph" || key == "a") return Alpha;
	else if (key == "nsteps" || key == "time" || key == "n") return Nsteps;
	else if (key == "dt") return Dt;
	else if (key == "iterperframe" || key == "framerate") return FrameRate;
	else if (key == "cxxxx") return Cxxxx;
	else if (key == "cxxyy") return Cxxyy;
	else if (key == "cxyxy") return Cxyxy;
	else if (key == "density" || key == "rho") return Density;
	else if (key == "damp" || key == "nu") return Damp;
	else if (key == "tpb" || key == "threadsperblock") return ThreadsPerBlock;
	else if (key == "outputbase" || key == "output" || key == "o") return OutputBase;
	else if (key == "meshfile" || key == "mesh") return MeshFile;
	else if (key == "maxnoderank" || key == "maxrank") return NodeRankMax;
	else if (key == "meshscale" || key == "scale") return MeshScale;
	else if (key == "caching" || key == "cache") return MeshCaching;
	else if (key == "planartop") { flagType = true; return PlanarSideUp; }
	else if (key == "homeotop") { flagType = true; return HomeoSideUp; }
	else if (key == "aplitude" || key == "sqzdheight" || key == "sqzamp") return Amplitude;
	else if (key == "ratio" || key == "sqzdlenght" || key == "length" || key == "l") return Ratio;
	else if (key == "smax" || key == "u") return Smax;
	else if (key == "smin" || key == "d") return Smin;
	else if (key == "sinit" || key == "s0" || key == "s") return SInitial;
	else if (key == "sonrate" || key == "onrate" || key == "son") return SRateOn;
	else if (key == "soffrate" || key == "offrate" || key == "soff") return SRateOff;
	else if (key == "phi" || key == "p" || key == "incidentangle") return IncidentAngle;
	else if (key == "iterperillum" || key == "illumrate") return IterPerIllumRecalc;
	else if (key == "startexperiment" || key == "start") return StartExperiment;
	else if (key == "stopexperiment" || key == "stop") return StopExperiment;
	else return Unknown;
}

bool ParametersReader::CleanKey(string &key)
{
	if (key.empty())
	{
		return false;
	}

	// remove lead and trailing spaces and hyphens
	size_t p = key.find_first_not_of(" \t-");
    key.erase(0, p);
   
    p = key.find_last_not_of(" \t");
    if (string::npos != p)
    {
    	key.erase(p+1);
    }	
	
	// removed everything
	if(key.empty())
	{
		return false;
	}
	
	return true;
}

bool ParametersReader::StrToBool(const string& str)
{
	if (str == "yes" || 
		str == "Yes" || 
		str == "true" || 
		str == "True")
	{
		return true;
	}
	else return false;
}


///
/// PARSE RESULT CLASS
///

ParseResult::ParseResult()
{
	this->Status = ParseStatus::READY_TO_PARSE;
}

ParseResult::ParseResult(ParseStatus status)
{
	this->Status = status;
}

ParseResult::ParseResult(ParseStatus status, string message)
{
	this->Status = status;
	this->Message = message;
}

ParseResult::ParseResult(const ParseResult& copy)
{
	this->Message = copy.Message;
	this->Status = copy.Status;
}

ParseResult& ParseResult::operator=(const ParseResult& rhs)
{
	this->Status = rhs.Status;
	this->Message = rhs.Message;
	return (*this);
}