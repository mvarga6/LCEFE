#ifndef __PRINTDIRECTOR_H__
#define __PRINTDIRECTOR_H__

void printDirector(real *THETA
				 , real *PHI
				 , int inZ
				 , int label
				 , std::string outputBase){

//print to file to test
real nx,ny,nz,th,ph;

char fout[128];
//sprintf(fout,"Output//director%d.xyzv",label);
sprintf(fout,"%s_director%d.xyzv", outputBase.c_str(),label);
FILE* locOut;
locOut = fopen(fout,"w");
fprintf(locOut,"%d\n",inX*inY*inZ);
fprintf(locOut,"#\n");

for(int i=0;i<inX;i++){
	for(int j=0;j<inY;j++){
		for(int k=0;k<inZ;k++){
			
			th = THETA[(i*inY+ j)*inZ+k];
			ph = PHI[(i*inY+ j)*inZ+k];

			nx = sin(th)*cos(ph);
			ny = sin(th)*sin(ph);
			nz = cos(th);

			fprintf(locOut,"A %d %d %d %f %f %f\n",i,j,k,nx,ny,nz);

		}//k
	}//j
}//i

}//printDirector

#endif//__PRINTDIRECTOR_H__
