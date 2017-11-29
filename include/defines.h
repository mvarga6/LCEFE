
#ifndef __DEFINES_H__
#define __DEFINES_H__

/// Uncomment to get explicit 
/// debugging information from
/// within kenel execution
///#define __DEBUG_FORCE__ 		10
///#define __DEBUG_SEND_FORCE__	20
///#define __DEBUG_SUM_FORCE__ 	10
///#define __DEBUG_UPDATE_V__ 	10
///#define __DEBUG_UPDATE_R__  	10

/// Changes the data precision
/// WARNING: Only works with float
#define real float

// Current workaround for DT of
// simulation.  float precision isn't
// good enough for dt^2 in update
//#define DT 0.0001

#endif
