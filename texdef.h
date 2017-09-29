#ifndef __TEXDEF_H__
#define __TEXDEF_H__

//#include "mainhead.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

extern size_t global_texture_offset;

//texture refference deffinitions
extern texture<float, 2, cudaReadModeElementType> texRef_r0;
extern texture<float, 2, cudaReadModeElementType> texRef_r;

#endif //__TEXDEF_H__
