#include "extlib/cxxopts/src/cxxopts.hpp"
#include "parameters.h"

void parseCommandLine(int argc, char* argv[]){

	cxxopts::Options inputs("GAFE6", "Finite element simulations of liquid crystal elastomers");
	inputs.add_options()
	("p,phi", 	"Set incident light angle (degrees)", 	cxxopts::value<float>()->default_value(85.0f))
	("a,alpha", 	"Set incident light intensity", 	cxxopts::value<float>()->default_value(5.0f))
	("t,smax",	"Set maximum value of order parameter", cxxopts::value<float>()->default_value(0.0f))
	("b,smin",	"Set minumum value of order parameter", cxxopts::value<float>()->default_value(-0.7f))
	("l,sqzdlength","Length of system post x-squeeze",	cxxopts::value<float>()->default_value(0.925f))
	("h,sqzdheight","Height of system post x-squeeze",	cxxopts::value<float>()->default_value(0.15f));

	inputs.parse(argc, argv);
	IANGLE = inputs["phi"].as<float>();
	SMAX = inputs["smax"].as<float>();
	SMIN = inputs["smin"].as<float>();
	SQZAMP = inputs["sqzdheight"].as<float>();
	SQZRATIO = inputs["sqzdlength"].as<float>();
}
