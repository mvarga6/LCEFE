//#include "extlib/cxxopts/src/cxxopts.hpp"
#include "parameters.h"
#include <string>
#include <cstdlib>

void printOptions();

// Cannot figure out proper build configuration to use extlib/cxxopts.hpp
/*void parseCommandLine(int argc, char* argv[]){

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
}*/


void parseCommandLine(int argc, char* argv[]){
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		std::string val;
		if (arg == "-h" || arg == "--help"){
			printOptions();
			exit(0);
		}
		else if (arg == "-p" || arg == "--phi"){
			if (i + 1 < argc) IANGLE = std::strtof(argv[1 + i++], NULL);
			else printf("\nOption '-p,--phi' requires one parameter.");
		}
		else if (arg == "-a" || arg == "--alpha"){

		}
		else if (arg == "-t" || arg == "--smax"){
			if (i + 1 < argc) SMAX = std::strtof(argv[1 + i++], NULL);
			else printf("\nOption '-t,--smax' requires one parameter.");
		}
		else if (arg == "-b" || arg == "--smin"){
			if (i + 1 < argc) SMIN = std::strtof(argv[1 + i++], NULL);
			else printf("\nOption '-b,--smin' requires one parameter.");
		}
		else if (arg == "-l" || arg == "--sqzdlength"){
			if (i + 1 < argc) SQZRATIO = std::strtof(argv[1 + i++], NULL);
			else printf("\nOption '-l,--sqzdlength' requires one parameter.");
		}
		else if (arg == "-H" || arg == "--sqzdheight"){
			if (i + 1 < argc) SQZAMP = std::strtof(argv[1 + i++], NULL);
			else printf("\nOption '-h,--sqzdheight' requires one parameter.");
		}
		else if (arg == "-o" || arg == "--output"){
			if (i + 1 < argc) OUTPUT.assign(argv[1 + i++]);
			else printf("\nOption '-o,--output' requires one parameter."); 
		}
		else if (arg == "-P" || arg == "--planartop"){
			PLANARTOP = true;
		}
		else if (arg == "-T" || arg == "--homeotop"){
			PLANARTOP = false;
		}
		else if (arg == "-r" || arg == "--onrate"){
			if (i + 1 < argc) SRATE_ON = std::strtof(argv[1 + i++], NULL);
			else printf("\nOption '-r,--onrate' requires one parameter.");
		}
		else if (arg == "-f" || arg == "--offrate"){
			if(i + 1 < argc) SRATE_OFF = std::strtof(argv[1 + i++], NULL);
			else printf("\nOption '-f,--offrate' requires one parameter.");
		}
		else printf("\nUnknown option: %s", arg.c_str());
	}
}


void printOptions(){
	printf("\n\n---------------------------------------------------------------------");
	printf("\n   GAFE6 -->  LIQUID CRYSTAL ELASTOMER FINITE ELEMENT SIMULATIONS  ");
	printf("\n---------------------------------------------------------------------");
	printf("\n\n  -p, --phi .............. Set incident light angle.");
	printf("\n  -a, --alpha ............ Set nematic order elastic coupling strenght.");
	printf("\n  -t, --smax ............. Set maximum value of order parameter.");
	printf("\n  -b, --smin ............. Set minimum value of order parameter.");
	printf("\n  -l, --sqzdlength ....... Length of mesh after x-squeeze.");
	printf("\n  -H, --sqzdheight ....... Height of mesh after x-squeeze.");
	printf("\n  -o, --output ........... VTK output of simulation.\n\n");
	printf("\n  -r, --onrate ........... Set dS/dt when illuminated.\n\n");
	printf("\n  -f, --offrate .......... Set dS/dt when in shadow.\n\n");
}
