
#ifndef __DEFINES_H__
#define __DEFINES_H__

///
/// Everything defined in this file needs to be set
/// when 'building' ('compiling') the code using 'make'
///

/// Uncomment to get explicit 
/// debugging information from
/// within kenel execution
#define __DEBUG_FORCE__ 		1000
//#define __DEBUG_READ_GLOBAL_MEMORY__ 10
// #define __DEBUG_SEND_FORCE__	20
// #define __DEBUG_SUM_FORCE__ 	10
// #define __DEBUG_UPDATE_V__ 	10
// #define __DEBUG_UPDATE_R__  	10

///
/// Data specifications
///
#define real float
#define MaxNodeRank 90
#define DefaultThreadsPerBlock 256

///
/// Math constants
///
#define DEG2RAD		0.017453293
#define PI		3.14159265359
#define _2PI	6.28318530718

#endif
