#ifndef __CL_PARSE_H__
#define __CL_PARSE_H__

//#include "extlib/cxxopts/src/cxxopts.hpp"


class SimulationParameters;

void printOptions();

void parseCommandLine(int argc, char* argv[], SimulationParameters *params);

#endif
