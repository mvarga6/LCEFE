#ifndef __GPUINFO_H__
#define __GPUINFO_H__



//Display information about GPU
void displayGPUinfo(cudaDeviceProp prop){
printf( " ---            General Information        ---\n\n");
printf( "Name: %s\n", prop.name );
printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
printf( "Clock rate: %d\n", prop.clockRate );
printf( "Device copy overlap: " );
if (prop.deviceOverlap)
printf( "Enabled\n" );
else
printf( "Disabled\n" );
printf( "Kernel execition timeout : " );
if (prop.kernelExecTimeoutEnabled)
printf( "Enabled\n" );
else
printf( "Disabled\n" );
printf( " \n\n---           Memory Information           ---\n\n");
printf( "Total global mem: %ld\n", prop.totalGlobalMem );
printf( "Total constant Mem: %ld\n", prop.totalConstMem );
printf( "Max mem pitch: %ld\n", prop.memPitch );
printf( "Texture Alignment: %ld\n", prop.textureAlignment );
printf( "Max 1D Textures %d\n",prop.maxTexture1D);
printf( "Max 2D Textures (%d,%d)\n",prop.maxTexture2D[0],prop.maxTexture2D[1]);
printf( "\n\n---         MP Information for device      ---\n\n");
printf( "Multiprocessor count: %d\n",
prop.multiProcessorCount );
printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
printf( "Registers per mp: %d\n", prop.regsPerBlock );
printf( "Threads in warp: %d\n", prop.warpSize );
printf( "Max threads per block: %d\n",
prop.maxThreadsPerBlock );
printf( "Max thread dimensions: (%d, %d, %d)\n",
prop.maxThreadsDim[0], prop.maxThreadsDim[1],
prop.maxThreadsDim[2] );
printf( "Max grid dimensions: (%d, %d, %d)\n",
prop.maxGridSize[0], prop.maxGridSize[1],
prop.maxGridSize[2] );
printf( "\n" );
}


#endif  // __GPUINFO_H__