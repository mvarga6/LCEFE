#ifndef GENRAND_H_
#define GENRAND_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MT_LEN          624
#define MT_IA           397
#define MT_IB           (MT_LEN - MT_IA)
#define UPPER_MASK      0x80000000
#define LOWER_MASK      0x7FFFFFFF
#define MATRIX_A        0x9908B0DF
#define TWIST(b,i,j)    ((b)[i] & UPPER_MASK) | ((b)[j] & LOWER_MASK)
#define MAGIC(s)        (((s)&1)*MATRIX_A)



void mt_init();
unsigned long randmt();
double genrand();
void purge();
double randgauss(double sigma, double mean);

#endif

