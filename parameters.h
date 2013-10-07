#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

//standard simulation parameters
#define NSTEPS			1000					    //total number of iterations
#define dt              0.000001					    //timestep [s]
#define iterPerFrame    5000                            //iterations per printed frame

//meshfile
#define MESHFILE "Mesh//beam100_100_100_n30399_e165931_f8670.dat"

//convert mesh length scale to cm
#define meshScale        0.1         //--[ cm / mesh unit]                      


//Elasticity constants (Lame' Coefficients)
//  -there is a factor of two off here from 
//  -the physical values, not sure if it
//  -is a half or two... I will find out
#define cxxxx			 29400000.0	  //--[ g / cm * s^2 ]			
#define cxxyy			 28000000.0	  //--[ g / cm * s^2 ]			
#define cxyxy			 570000.0	  //--[ g / cm * s^2 ]	


//Q:elasticity coupling constant
#define alpha            570000.0*1.5 //--[ g / cm * s^2 ]


//Density of elastomer material
#define materialDensity  1.2   //--[ g / cm^3 ]  =  [  10^3 kg / m^3 ]


//scalar velocity dampening
//each velocity multiplied by this at each step
#define damp             0.999        //--[ unitless ]


//x and y dimensions of n profile
//input arrays 
#define inX 200
#define inY 200

//Threads per block to exicute
//100 seems pretty optimal on GTX275
//might be better with larger on 
//better card/ differnt card
#define TPB				100			



//maximum number of tetrahedra
//a Node can belone to
#define MaxNodeRank     90							



//constants declared on the stack for speed
#define PI				3.14159265359
#define dt2o2           (dt*dt)/2.0					    //for speed
#define dto2             dt/2.0						    //for speed

#endif //__PARAMETERS_H__
