#include "clparse.h"
#include <stdio.h>
#include <string>
#include <cstdlib>
#include "simulation_parameters.h"

void parseCommandLine(int argc, char* argv[], SimulationParameters *params){
	
	
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		std::string val;
		if (arg == "-h" || arg == "--help")
		{
			printOptions();
			exit(0);
		}
		else if (arg == "-p" || arg == "--phi")
		{
			if (i + 1 < argc) 
			{
				float IANGLE = std::strtof(argv[1 + i++], NULL);
				params->Actuation.Optics.IncidentAngle = IANGLE;
			}
			else printf("\nOption '-p,--phi' requires one parameter.");
			
		}
		else if (arg == "-a" || arg == "--alpha")
		{
			if (i + 1 < argc) 
			{
				params->Material.Alpha = std::strtof(argv[1 + i++], NULL);
			}
			else printf("\nOption '-a,--alpha' requires one parameter.");
		}
		else if (arg == "-t" || arg == "--smax")
		{
			if (i + 1 < argc) 
			{
				float SMAX = std::strtof(argv[1 + i++], NULL);
				params->Actuation.OrderParameter.Smax = SMAX;
			}
			else printf("\nOption '-t,--smax' requires one parameter.");
		}
		else if (arg == "-b" || arg == "--smin")
		{
			if (i + 1 < argc)
			{
				float SMIN = std::strtof(argv[1 + i++], NULL);
				params->Actuation.OrderParameter.Smin = SMIN;
			} 
			else printf("\nOption '-b,--smin' requires one parameter.");
		}
		else if (arg == "-l" || arg == "--sqzdlength")
		{
			if (i + 1 < argc) 
			{
				float SQZRATIO = std::strtof(argv[1 + i++], NULL);
				params->Initalize.SqueezeRatio = SQZRATIO;
			}
			else printf("\nOption '-l,--sqzdlength' requires one parameter.");
		}
		else if (arg == "-H" || arg == "--sqzdheight")
		{
			if (i + 1 < argc) 
			{
				float SQZAMP = std::strtof(argv[1 + i++], NULL);
				params->Initalize.SqueezeAmplitude = SQZAMP;
			}
			else printf("\nOption '-H,--sqzdheight' requires one parameter.");
		}
		else if (arg == "-o" || arg == "--output")
		{
			if (i + 1 < argc) 
			{
				std::string OUTPUT(argv[1 + i++]);
				params->Output.Base = OUTPUT;
			}
			else printf("\nOption '-o,--output' requires one parameter."); 
		}
		else if (arg == "-P" || arg == "--planartop")
		{
			bool PLANARTOP = true;
			params->Initalize.PlanarSideUp = true;
		}
		else if (arg == "-T" || arg == "--homeotop")
		{
			bool PLANARTOP = false;
			params->Initalize.PlanarSideUp = false;
		}
		else if (arg == "-r" || arg == "--onrate")
		{
			if (i + 1 < argc) 
			{
				float SRATE_ON = std::strtof(argv[1 + i++], NULL);
				params->Actuation.OrderParameter.SRateOn = SRATE_ON;
			}
			else printf("\nOption '-r,--onrate' requires one parameter.");
		}
		else if (arg == "-f" || arg == "--offrate")
		{
			if(i + 1 < argc) 
			{
				float SRATE_OFF = std::strtof(argv[1 + i++], NULL);
				params->Actuation.OrderParameter.SRateOff = SRATE_OFF;
			}
			else printf("\nOption '-f,--offrate' requires one parameter.");
		}
		else printf("\nUnknown option: %s", arg.c_str());
	}
}


void printOptions(){
	printf("\n\n---------------------------------------------------------------------");
	printf("\n   GAFE6 -->  LIQUID CRYSTAL ELASTOMER FINITE ELEMENT SIMULATIONS  ");
	printf("\n---------------------------------------------------------------------\n");
	printf("\n  -p, --phi .............. Set incident light angle.");
	printf("\n  -a, --alpha ............ Set nematic order elastic coupling strenght.");
	printf("\n  -t, --smax ............. Set maximum value of order parameter.");
	printf("\n  -b, --smin ............. Set minimum value of order parameter.");
	printf("\n  -l, --sqzdlength ....... Length of mesh after x-squeeze.");
	printf("\n  -H, --sqzdheight ....... Height of mesh after x-squeeze.");
	printf("\n  -o, --output ........... VTK output of simulation.\n\n");
	printf("\n  -r, --onrate ........... Set dS/dt when illuminated.\n\n");
	printf("\n  -f, --offrate .......... Set dS/dt when in shadow.\n\n");
}
