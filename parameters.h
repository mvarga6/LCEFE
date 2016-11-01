#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

//standard simulation parameters
#define NSTEPS		500000		   //total number of iterations
#define dt              0.00001	   //timestep [s]
//#define dt              0.0000005	   //timestep [s]
#define iterPerFrame    2000              //iterations per printed frame

//meshfile
#define MESHFILE "Mesh/beam400_50_50_n1566_e6037_f2022.dat"
//#define MESHFILE "Mesh/lcefilm2.msh"

//convert mesh length scale to cm
#define meshScale        0.1         //--[ cm / mesh unit]

//parameters regarding sample illumination
#define SRES		1000	     //--resolution for float to int conversion for gpu storage
#define S0		1.0	     //--naturally occuring order parameter (no illuminiation)
#define LRATE		10

//Elasticity constants (Lame' Coefficients)
//  -there is a factor of two off here from
//  -the physical values, not sure if it
//  -is a half or two... I will find out
#define cxxxx		29400000.0	  //--[ g / cm * s^2 ]
#define cxxyy		28000000.0	  //--[ g / cm * s^2 ]
#define cxyxy		570000.0	  //--[ g / cm * s^2 ]


//Q:elasticity coupling constant
#define alpha            570000.0*1.5 //--[ g / cm * s^2 ]


//Density of elastomer material
#define materialDensity  1.2  		//--[ g / cm^3 ]  =  [  10^3 kg / m^3 ]


//scalar velocity dampening
//each velocity multiplied by this at each step
#define damp            0.999		//1.0=no damping      //--[ unitless ]


//x and y dimensions of n profile
//input arrays
#define inX 200
#define inY 200

//Threads per block to exicute
//100 seems pretty optimal on GTX275
//might be better with larger on
//better card/ differnt card
#define TPB		256



//maximum number of tetrahedra
//a Node can belone to
#define MaxNodeRank     90



//constants declared on the stack for speed
#define PI		3.14159265359
#define dt2o2           (dt*dt)/2.0					    //for speed
#define dto2             dt/2.0						    //for speed
#define DEG2RAD		0.017453293

#endif //__PARAMETERS_H__
