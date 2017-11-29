#ifndef GENRAND_H_
#define GENRAND_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

///
/// Classic C-style random number generator module
///

///
/// Definitions for random number generation
#define MT_LEN          624
#define MT_IA           397
#define MT_IB           (MT_LEN - MT_IA)
#define UPPER_MASK      0x80000000
#define LOWER_MASK      0x7FFFFFFF
#define MATRIX_A        0x9908B0DF
#define TWIST(b,i,j)    ((b)[i] & UPPER_MASK) | ((b)[j] & LOWER_MASK)
#define MAGIC(s)        (((s)&1)*MATRIX_A)

///
/// initialize the components
void mt_init();

///
/// Get a random unsigned long [0, MAX)
unsigned long randmt();

///
/// Get a random double [0, 1)
double genrand();

///
/// Purges the random seed
void purge();

///
/// Get a random double from a Gaussian distribution
double randgauss(double sigma, double mean);

#endif

