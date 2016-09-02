#ifndef __ILLUMINATION_H__
#define __ILLUMINATION_H__
#include "mainhead.h"
#include "parameters.h"
#include <math.h>

__device__ void illumination(float (&r)[12]
				,int (&node_num)[4]
				,int *TetToNode
				,int TTNshift
				,int nodeRanks[4]
				,float &last_illum_time
				,float &dt_since_illum
				,float (&light_src)[3]
				,float t
				){

	// if not a surface tetra stop
	
	// calculate ray to light source

	// if ray passes through another surface 
	// tetra (not currently illuminated) calc
	// time since illuminated. calc by looping
	//
	// dt_since_illum = t - last_illum_time
	//
	// else dt_since_illum = 0
}
