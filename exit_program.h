#ifndef __EXIT_PROGRAM_H__
#define __EXIT_PROGRAM_H__


void exit_program(DevDataBlock *dev_dat){
cudaUnbindTexture( texRef_r0 );
cudaUnbindTexture( texRef_r );



HANDLE_ERROR( cudaFree( dev_dat->dev_A ) );
HANDLE_ERROR( cudaFree( dev_dat->dev_TetToNode ) );
HANDLE_ERROR( cudaFree( dev_dat->dev_r0 ) );
HANDLE_ERROR( cudaFree( dev_dat->dev_r ) );
HANDLE_ERROR( cudaFree( dev_dat->dev_F ) );
HANDLE_ERROR( cudaFree( dev_dat->dev_v ) );
}



#endif//__EXIT_PROGRAM_H__