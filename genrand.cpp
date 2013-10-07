#include "genrand.h"

int mt_index;
unsigned long mt_buffer[MT_LEN];

/**********************************************************************/
/** All function bodies for random number generator using MT         **/
/**********************************************************************/

void mt_init() {
    int i;
    for (i = 0; i < MT_LEN; i++)
        mt_buffer[i] = rand();
    mt_index = 0;
}



unsigned long randmt() {
    unsigned long * b = mt_buffer;
    int idx = mt_index;
    unsigned long s;
    int i;
	
    if (idx == MT_LEN*sizeof(unsigned long))
    {
        idx = 0;
        i = 0;
        for (; i < MT_IB; i++) {
            s = TWIST(b, i, i+1);
            b[i] = b[i + MT_IA] ^ (s >> 1) ^ MAGIC(s);
        }
        for (; i < MT_LEN-1; i++) {
            s = TWIST(b, i, i+1);
            b[i] = b[i - MT_IB] ^ (s >> 1) ^ MAGIC(s);
        }
        
        s = TWIST(b, MT_LEN-1, 0);
        b[MT_LEN-1] = b[MT_IA-1] ^ (s >> 1) ^ MAGIC(s);
    }
    mt_index = idx + sizeof(unsigned long);
    return *(unsigned long *)((unsigned char *)b + idx);
}
double genrand()
{
unsigned long b=randmt();
(double)b;
double a;
a=b*0.000000000232830644;
return a;
}
void purge()
{
for (int j = 0; j < 2500; j++)
	{
		genrand();
	}
}

/**********************************************************************/
/** Generates numbers from a normal distrobution                     **/
/**********************************************************************/

double randgauss(double sigma, double mean)
{
float x1, x2, w, y1, y2, gauss;
 
         do {
                 x1 = 2.0 * genrand() - 1.0;
                 x2 = 2.0 * genrand() - 1.0;
                 w = x1 * x1 + x2 * x2;
         } while ( w >= 1.0 );

         w = sqrt( (-2.0 * log( w ) ) / w );
         y1 = x1 * w;
         y2 = x2 * w;
		gauss = y2*sigma + mean;
return gauss;
}


