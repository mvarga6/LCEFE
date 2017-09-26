#ifndef __PRINTMESHORDERING_H__
#define __PRINTMESHORDERING_H__

void printorder(TetArray &Tets, std::string outputBase)
{
	std::string fileName = outputBase + "_meshorder.xyzv";

	printf("\nWriting Tets order to %s\n", fileName.c_str());
	int Ntets = Tets.size;

	char fout[128];
	sprintf(fout, "%s", fileName.c_str());
	FILE*out1;
//	out1 =fopen("Output//meshorder.xyzv","w");
	out1 =fopen(fout,"w");
	fprintf(out1,"%d\n", Ntets);
	fprintf(out1,"#\n");

	for(int n = 0; n < Ntets - 1; n++){

		fprintf(out1,"A %f %f %f %f %f %f\n",Tets.get_pos(n,0) \
				,Tets.get_pos(n,1) \
				,Tets.get_pos(n,2) \
				,Tets.get_pos(n+1,0) - Tets.get_pos(n,0) \
				,Tets.get_pos(n+1,1) - Tets.get_pos(n,1) \
				,Tets.get_pos(n+1,2) - Tets.get_pos(n,2) );
	}

	fclose(out1);
 }


#endif //__PRINTMESHORDERING_H__
