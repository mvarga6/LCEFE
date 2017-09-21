#ifndef __EXIT_PROGRAM_H__
#define __EXIT_PROGRAM_H__


void exit_program(DevDataBlock *dev)
{
	cudaUnbindTexture( texRef_r0 );
	cudaUnbindTexture( texRef_r );

	HANDLE_ERROR( cudaFree( dev->A ) );
	HANDLE_ERROR( cudaFree( dev->TetToNode ) );
	HANDLE_ERROR( cudaFree( dev->r0 ) );
	HANDLE_ERROR( cudaFree( dev->r ) );
	HANDLE_ERROR( cudaFree( dev->F ) );
	HANDLE_ERROR( cudaFree( dev->v ) );
}

#endif//__EXIT_PROGRAM_H__
