#ifndef __ANYERRORS_H__
#define __ANYERRORS_H__

//let us know if any errors have happend

void any_errors(void){
	printf("\nErrors so far?\n");
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
	printf("Yes, damn! \n Here is what went wrong:\n");
    printf("Error: %s\n\n", cudaGetErrorString(err));
	}else{printf("No sir, You are awesome!\n\n");}
}

#endif//__ANYERRORS_H__