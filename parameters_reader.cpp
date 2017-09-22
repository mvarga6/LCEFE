
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

using namespace std;

ParseResult ParametersReader::ReadFromFile(const string& fileName, SimulationParameters& parameters)
{
	if (fileName.empty())
	{
		return MESHFILE_NAME_IS_NULL;
	}

	tokenMap map = ParseFileToTokenMap(fileName);
	ConvertTokenMapToParameters(map, parameters);
	return this->status;
}

ParseResult ParametersReader::ReadFromCmdline(int argc, char* argv[], SimulationParameters& parameters)
{
	if (argc < 1)
	{
		return ARGS_COUNT_ZERO;
	}
	
	if (argv == NULL)
	{
		return ARGS_MISSING;
	}
	
	tokenMap map = ParseCmdlineToTokenMap(argc, argv);
	ConvertTokenMapToParameters(map, parameters);
	return this->status;
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
	
	this->status = SUCCESS;
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
		this->status = CRITICAL_FAILURE;
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
		this->status = CRITICAL_FAILURE;
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
		this->status = CRITICAL_FAILURE;
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
				if (type != Unknown)
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
	
	this->status = SUCCESS;
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
		case Unknown: 	break;
		case ParametersFile: p.File = v; break;
		case Cxxxx: 	p.Material.Cxxxx = ::atof(v.c_str()); break;
		case Cxxyy: 	p.Material.Cxxyy = ::atof(v.c_str()); break;
		case Cxyxy: 	p.Material.Cxyxy = ::atof(v.c_str()); break;
		case Alpha: 	p.Material.Alpha = ::atof(v.c_str()); break;
		case Density: 	p.Material.Density = ::atof(v.c_str()); break;
		case Nsteps:	p.Dynamics.Nsteps = ::atoi(v.c_str()); break;
		case Dt: 		p.Dynamics.Dt = ::atof(v.c_str()); break;
		case Damp: 		p.Dynamics.Damp = ::atof(v.c_str()); break;
		case ThreadsPerBlock:		p.Gpu.ThreadsPerBlock = ::atoi(v.c_str()); break;
		case OutputBase: 	p.Output.Base = v; break;
		case FrameRate: 	p.Output.FrameRate = ::atoi(v.c_str()); break;
		case MeshFile:		p.Mesh.File = v; break;
		case NodeRankMax: 	p.Mesh.NodeRankMax = ::atoi(v.c_str()); break;
		case MeshScale: 	p.Mesh.Scale = ::atof(v.c_str()); break;
		case PlanarSideUp: 	p.Initalize.PlanarSideUp = true; break;
		case HomeoSideUp: 	p.Initalize.PlanarSideUp = false; break;
		case Amplitude: 	p.Initalize.SqueezeAmplitude = ::atof(v.c_str()); break;
		case Ratio: 		p.Initalize.SqueezeRatio = ::atof(v.c_str()); break;
		case SInitial: 		p.Actuation.OrderParameter.SInital = ::atof(v.c_str()); break;
		case Smax: 			p.Actuation.OrderParameter.Smax = ::atof(v.c_str()); break;
		case Smin: 			p.Actuation.OrderParameter.Smin = ::atof(v.c_str()); break;
		case SRateOn: 		p.Actuation.OrderParameter.SRateOn = ::atof(v.c_str()); break;
		case SRateOff: 		p.Actuation.OrderParameter.SRateOff = ::atof(v.c_str()); break;
		case IncidentAngle: p.Actuation.Optics.IncidentAngle = ::atof(v.c_str()); break;
		case IterPerIllumRecalc: p.Actuation.Optics.IterPerIllumRecalc = ::atoi(v.c_str()); break;
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
		return Unknown;
	}

	if (key == "input" || key == "params" || key == "parameters") return ParametersFile;
	if (key == "alpha" || key == "alph" || key == "a") return Alpha;
	if (key == "nsteps" || key == "time" || key == "n") return Nsteps;
	if (key == "dt") return Dt;
	if (key == "iterperframe" || key == "framerate") return FrameRate;
	if (key == "cxxxx") return Cxxxx;
	if (key == "cxxyy") return Cxxyy;
	if (key == "cxyxy") return Cxyxy;
	if (key == "density" || key == "rho") return Density;
	if (key == "damp" || key == "nu") return Damp;
	if (key == "tpb" || key == "threadsperblock") return ThreadsPerBlock;
	if (key == "outputbase" || key == "output" || key == "o") return OutputBase;
	if (key == "meshfile" || key == "mesh") return MeshFile;
	if (key == "maxnoderank" || key == "maxrank") return NodeRankMax;
	if (key == "meshscale" || key == "scale") return MeshScale;
	if (key == "planartop") { flagType = true; return PlanarSideUp; }
	if (key == "homeotop") { flagType = true; return HomeoSideUp; }
	if (key == "aplitude" || key == "sqzdheight" || key == "sqzamp") return Amplitude;
	if (key == "ratio" || key == "sqzdlenght" || key == "length" || key == "l") return Ratio;
	if (key == "smax" || key == "u") return Smax;
	if (key == "smin" || key == "d") return Smin;
	if (key == "sinit" || key == "s0" || key == "s") return SInitial;
	if (key == "sonrate" || key == "onrate" || key == "son") return SRateOn;
	if (key == "soffrate" || key == "offrate" || key == "soff") return SRateOff;
	if (key == "phi" || key == "p" || key == "incidentangle") return IncidentAngle;
	if (key == "iterperillum" || key == "illumrate") return IterPerIllumRecalc;
	return Unknown;
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

//		if (arg == "-h" || arg == "--help")
//		{
//			//printOptions();
//			exit(0);
//		}
//		else if (arg == "-p" || arg == "--phi")
//		{
//			if (i + 1 < argc) 
//			{
//				pairs["phi"] = string(argv[1 + i++]);
//			}
//		}
//		else if (arg == "-a" || arg == "--alpha")
//		{
//			if (i + 1 < argc) 
//			{
//				pairs["alpha"] = string(argv[1 + i++]);
//			}
//		}
//		else if (arg == "-t" || arg == "--smax")
//		{
//			if (i + 1 < argc) 
//			{
//				pairs["smax"] = string(argv[1 + i++]);
//			}
//		}
//		else if (arg == "-b" || arg == "--smin")
//		{
//			if (i + 1 < argc)
//			{
//				pairs["smin"] = string(argv[1 + i++]);
//			} 
//		}
//		else if (arg == "-l" || arg == "--sqzdlength")
//		{
//			if (i + 1 < argc) 
//			{
//				pairs["prebend"] = string(argv[1 + i++]);
//			}
//		}
//		else if (arg == "-H" || arg == "--sqzdheight")
//		{
//			if (i + 1 < argc) 
//			{
//				pairs["prebendHeight"] = string(argv[1 + i++]);
//			}
//		}
//		else if (arg == "-o" || arg == "--output")
//		{
//			if (i + 1 < argc) 
//			{
//				pairs["meshFile"] = string(argv[1 + i++]);
//			}
//		}
//		else if (arg == "-P" || arg == "--planartop")
//		{
//			pairs["planartop"] = "true"
//		}
//		else if (arg == "-T" || arg == "--homeotop")
//		{
//			pairs["planartop"] = "false"
//		}
//		else if (arg == "-r" || arg == "--onrate")
//		{
//			if (i + 1 < argc) 
//			{
//				pairs["onRate"] = string(argv[1 + i++]);
//			}
//		}
//		else if (arg == "-f" || arg == "--offrate")
//		{
//			if(i + 1 < argc) 
//			{
//				pairs["offRate"] = string(argv[1 + i++]);
//			}
//		}
//		else printf("\nUnknown option: %s", arg.c_str());
//	}

