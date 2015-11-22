#ifndef __EXIT_PROGRAM_H__
#define __EXIT_PROGRAM_H__


void exit_program(DevDataBlock *dev_dat){
	cudaUnbindTexture(texRef_r0);
	cudaUnbindTexture(texRef_r);
	HANDLE_ERROR(cudaFree(dev_dat->dev_A));
	HANDLE_ERROR(cudaFree(dev_dat->dev_TetToNode));
	HANDLE_ERROR(cudaFree(dev_dat->dev_r0));
	HANDLE_ERROR(cudaFree(dev_dat->dev_r));
	HANDLE_ERROR(cudaFree(dev_dat->dev_F));
	HANDLE_ERROR(cudaFree(dev_dat->dev_v));
	HANDLE_ERROR(cudaFree(dev_dat->dev_nodeRank));
	HANDLE_ERROR(cudaFree(dev_dat->dev_TetNodeRank));
	HANDLE_ERROR(cudaFree(dev_dat->dev_dr));
	HANDLE_ERROR(cudaFree(dev_dat->dev_m));
	HANDLE_ERROR(cudaFree(dev_dat->dev_pe));
	HANDLE_ERROR(cudaFree(dev_dat->dev_TetVol));
	HANDLE_ERROR(cudaFree(dev_dat->dev_ThPhi));
}

#endif//__EXIT_PROGRAM_H__