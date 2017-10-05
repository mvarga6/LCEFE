
#include "parameters_reader.h"
#include "simulation_parameters.h"
//#include "clparse.h"
#include "extlib/jsmn/jsmn.h"
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "logger.h"

using namespace std;

ParseResult ParametersReader::ReadFromFile(const string& fileName, SimulationParameters& parameters)
{
	if (fileName.empty())
	{
		return ParseResult::MESHFILE_NAME_IS_NULL;
	}

	tokenMap map = ParseFileToTokenMap(fileName);
	ConvertTokenMapToParameters(map, parameters);
	return this->status;
}

ParseResult ParametersReader::ReadFromCmdline(int argc, char* argv[], SimulationParameters& parameters)
{
	if (argc < 1)
	{
		return ParseResult::ARGS_COUNT_ZERO;
	}
	
	if (argv == NULL)
	{
		return ParseResult::ARGS_MISSING;
	}
	
	tokenMap map = ParseCmdlineToTokenMap(argc, argv);
	ConvertTokenMapToParameters(map, parameters);
	return this->status;
}


string ParametersReader::GetErrorMsg(ParseResult result)
{
	switch(result)
	{
	case ParseResult::SUCCESS: return "No Error";
	case ParseResult::CRITICAL_FAILURE: return "Unspecified Critical Error";
	case ParseResult::MESHFILE_NAME_IS_NULL: return "Mesh file name is not provided.  Use 'meshfile' in parameters file or cmdline flags.";
	case ParseResult::ARGS_COUNT_ZERO: return "No Options Provided.  Requires at least 'input' for parameters file.";
	case ParseResult::ARGS_MISSING: return "Option Missing Parameter";
	default: return "Unknown";
	}
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
		if (type != ParameterType::Unknown)
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
	
	this->status = ParseResult::SUCCESS;
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
		this->status = ParseResult::CRITICAL_FAILURE;
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
	if (!HandleParseResult(num_tokens))
	{
		this->status = ParseResult::CRITICAL_FAILURE;
		return pairs;
	}
	
	// create tokens array and reset parser
	jsmntok_t *tokens = new jsmntok_t[num_tokens];
	jsmn_init(&parser);
	
	// actually read the tokens now
	int result = jsmn_parse(&parser, js, strlen(js), tokens, num_tokens);
	
	// check result
	if (!HandleParseResult(result))
	{
		this->status = ParseResult::CRITICAL_FAILURE;
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
				if (type != ParameterType::Unknown)
				{
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
	}
	
	fin.close();
	
	this->status = ParseResult::SUCCESS;
	return pairs;
}


void ParametersReader::ConvertTokenMapToParameters(tokenMap &map, SimulationParameters &p)
{
	tokenMapIterator begin = map.begin();
	tokenMapIterator end = map.end();
	tokenMapIterator it;
	
	for(it = begin; it != end; it++)
	{
		string v = it->second;
	
		switch(it->first)
		{
		case ParameterType::Unknown: 	break;
		case ParameterType::ParametersFile: p.File = v; break;
		case ParameterType::Cxxxx: 	p.Material.Cxxxx = ::atof(v.c_str()); break;
		case ParameterType::Cxxyy: 	p.Material.Cxxyy = ::atof(v.c_str()); break;
		case ParameterType::Cxyxy: 	p.Material.Cxyxy = ::atof(v.c_str()); break;
		case ParameterType::Alpha: 	p.Material.Alpha = ::atof(v.c_str()); break;
		case ParameterType::Density: 	p.Material.Density = ::atof(v.c_str()); break;
		case ParameterType::Nsteps:	p.Dynamics.Nsteps = ::atoi(v.c_str()); break;
		case ParameterType::Dt: 		p.Dynamics.Dt = ::atof(v.c_str()); break;
		case ParameterType::Damp: 		p.Dynamics.Damp = ::atof(v.c_str()); break;
		case ParameterType::ThreadsPerBlock:		p.Gpu.ThreadsPerBlock = ::atoi(v.c_str()); break;
		case ParameterType::OutputBase: 	p.Output.Base = v; break;
		case ParameterType::FrameRate: 	p.Output.FrameRate = ::atoi(v.c_str()); break;
		case ParameterType::MeshFile:		p.Mesh.File = v; break;
		case ParameterType::NodeRankMax: 	p.Mesh.NodeRankMax = ::atoi(v.c_str()); break;
		case ParameterType::MeshScale: 	p.Mesh.Scale = ::atof(v.c_str()); break;
		case ParameterType::MeshCaching:	p.Mesh.CachingOn = StrToBool(v); break;
		case ParameterType::PlanarSideUp: 	p.Initalize.PlanarSideUp = true; break;
		case ParameterType::HomeoSideUp: 	p.Initalize.PlanarSideUp = false; break;
		case ParameterType::Amplitude: 	p.Initalize.SqueezeAmplitude = ::atof(v.c_str()); break;
		case ParameterType::Ratio: 		p.Initalize.SqueezeRatio = ::atof(v.c_str()); break;
		case ParameterType::SInitial: 		p.Actuation.OrderParameter.SInitial = ::atof(v.c_str()); break;
		case ParameterType::Smax: 			p.Actuation.OrderParameter.Smax = ::atof(v.c_str()); break;
		case ParameterType::Smin: 			p.Actuation.OrderParameter.Smin = ::atof(v.c_str()); break;
		case ParameterType::SRateOn: 		p.Actuation.OrderParameter.SRateOn = ::atof(v.c_str()); break;
		case ParameterType::SRateOff: 		p.Actuation.OrderParameter.SRateOff = ::atof(v.c_str()); break;
		case ParameterType::IncidentAngle: p.Actuation.Optics.IncidentAngle = ::atof(v.c_str()); break;
		case ParameterType::IterPerIllumRecalc: p.Actuation.Optics.IterPerIllumRecalc = ::atoi(v.c_str()); break;
		case ParameterType::LoggerType: 	p.Output.LogType = ConvertToLoggerType(::atoi(v.c_str())); break;
		case ParameterType::LoggerLevel:	p.Output.LogLevel = ConvertToLogEntryPriority(::atoi(v.c_str())); break;
		default: break;
		}
	} 
}


bool ParametersReader::HandleParseResult(int result)
{
	switch(result)
	{
	case JSMN_ERROR_NOMEM:
		cout << "JSMN ERROR: Not enough tokens provided" << endl;
		return false;
		
	case JSMN_ERROR_INVAL:
		cout << "JSMN ERROR: Invalid character in parameters file" << endl;	
		return false;
		
	case JSMN_ERROR_PART:
		cout << "JSMN ERROR: Incomplete JSON packet, expected more bytes" << endl;
		return false;
		
	default:
		return true;
	}
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
		return ParameterType::Unknown;
	}
	
	if (key == "input" || key == "params" || key == "parameters") return ParameterType::ParametersFile;
	if (key == "alpha" || key == "alph" || key == "a") return ParameterType::Alpha;
	if (key == "nsteps" || key == "time" || key == "n") return ParameterType::Nsteps;
	if (key == "dt") return ParameterType::Dt;
	if (key == "iterperframe" || key == "framerate") return ParameterType::FrameRate;
	if (key == "cxxxx") return ParameterType::Cxxxx;
	if (key == "cxxyy") return ParameterType::Cxxyy;
	if (key == "cxyxy") return ParameterType::Cxyxy;
	if (key == "density" || key == "rho") return ParameterType::Density;
	if (key == "damp" || key == "nu") return ParameterType::Damp;
	if (key == "tpb" || key == "threadsperblock") return ParameterType::ThreadsPerBlock;
	if (key == "outputbase" || key == "output" || key == "o") return ParameterType::OutputBase;
	if (key == "meshfile" || key == "mesh") return ParameterType::MeshFile;
	if (key == "maxnoderank" || key == "maxrank") return ParameterType::NodeRankMax;
	if (key == "meshscale" || key == "scale") return ParameterType::MeshScale;
	if (key == "caching" || key == "cache") return ParameterType::MeshCaching;
	if (key == "planartop") { flagType = true; return ParameterType::PlanarSideUp; }
	if (key == "homeotop") { flagType = true; return ParameterType::HomeoSideUp; }
	if (key == "aplitude" || key == "sqzdheight" || key == "sqzamp") return ParameterType::Amplitude;
	if (key == "ratio" || key == "sqzdlenght" || key == "length" || key == "l") return ParameterType::Ratio;
	if (key == "smax" || key == "u") return ParameterType::Smax;
	if (key == "smin" || key == "d") return ParameterType::Smin;
	if (key == "sinit" || key == "s0" || key == "s") return ParameterType::SInitial;
	if (key == "sonrate" || key == "onrate" || key == "son") return ParameterType::SRateOn;
	if (key == "soffrate" || key == "offrate" || key == "soff") return ParameterType::SRateOff;
	if (key == "phi" || key == "p" || key == "incidentangle") return ParameterType::IncidentAngle;
	if (key == "iterperillum" || key == "illumrate") return ParameterType::IterPerIllumRecalc;
	if (key == "logtype" || key == "logger" || key == "logging") return ParameterType::LoggerType;
	if (key == "loglevel" || key == "level") return ParameterType::LoggerLevel;
	return ParameterType::Unknown;
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

