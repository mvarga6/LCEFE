#ifndef __PRINTMESHORDERING_H__
#define __PRINTMESHORDERING_H__

void printorder(TetArray &i_Tet,int Ntets){
	char fout[128];
	sprintf(fout, "%s_meshorder.xyzv", OUTPUT.c_str());
	FILE*out1;
//	out1 =fopen("Output//meshorder.xyzv","w");
	out1 =fopen(fout,"w");
	fprintf(out1,"%d\n",Ntets);
	fprintf(out1,"#\n");

	for(int n = 0 ;n<Ntets-1;n++){

		fprintf(out1,"A %f %f %f %f %f %f\n",i_Tet.get_pos(n,0) \
				,i_Tet.get_pos(n,1) \
				,i_Tet.get_pos(n,2) \
				,i_Tet.get_pos(n+1,0)-i_Tet.get_pos(n,0) \
				,i_Tet.get_pos(n+1,1)-i_Tet.get_pos(n,1) \
				,i_Tet.get_pos(n+1,2)-i_Tet.get_pos(n,2) );
	}

	fclose(out1);
 }


#endif //__PRINTMESHORDERING_H__
