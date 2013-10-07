#ifndef __DATASTRUCT_H__
#define __DATASTRUCT_H__



//  class to hold the tetrahedral array with instances which will be usefull for renumbering
class TetArray{

private:
	int *TetNab;
	float *TetPos;
	int size;


public:
	TetArray(int size);
	~TetArray();

	void set_nabs(int i, int j,const int &newval);
	void set_pos(int i, int j,const float &newval);
	int get_nab(int i, int j);
	float get_pos(int i, int j);
	void switch_tets(int i, int j);

};

TetArray::TetArray(int N){
	size = N;
	TetNab = new int[size*4];
	TetPos = new float[size*3];
}

TetArray::~TetArray(){
	delete TetNab;
	TetNab = NULL;
	delete TetPos;
	TetNab = NULL;
}

void TetArray::set_nabs(int i, int j,const int &newval){
	if(i<size && i>=0){
		if(j<4 && j>=0){
		TetNab[i*4+j] = newval;
		}else{
		printf("index out of bounds writing to TetNabs\n");
		}
	}else{
		printf("index out of bounds writing to TetNabs\n");
	}
}

void TetArray::set_pos(int i, int j,const float &newval){
	if(i<size && i>=0){
		if(j<3 && j>=0){
		TetPos[i*3+j] = newval;
		}else{
		printf("index out of bounds writing to TetPos\n");
		}
	}else{
		printf("index out of bounds writing to TetPos\n");
	}
}


int TetArray::get_nab(int i, int j){
	if(i<size && i>=0){
		if(j<4 && j>=0){
		return TetNab[i*4+j];
		}
		else{
		printf("index out of bounds reading from TetNabs\n");
		return 1000;
		}
	}else{
		printf("index out of bounds reading from TetNabs\n");
		return 1000;
	}
}

float TetArray::get_pos(int i, int j){
	if(i<size && i>=0){
		if(j<3 && j>=0){
		return TetPos[i*3+j];
		}else{
		printf("index out of bounds reading from TetPos\n");
		return 1000;
		}
	}else{
		printf("index out of bounds reading from TetPos\n");
		return 1000;
	}
}

//switch all elemnts of both Tet arrays for i and j
void TetArray::switch_tets(int i, int j){
	float buffpos;
	int buffnab;

	if(i<size && i>=0){
		if(j<size && j>=0){
			for(int p = 0;p<4;p++){
				if(p<3){
				buffpos = TetPos[i*3+p];
				TetPos[i*3+p] = TetPos[j*3+p];
				TetPos[j*3+p] = buffpos;
				}

				buffnab = TetNab[i*4+p];
				TetNab[i*4+p] = TetNab[j*4+p];
				TetNab[j*4+p] = buffnab;
			}
		
		}else{
		printf("index out of bounds switching Tetrahedra numbers\n");
		}
	}else{
		printf("index out of bounds switching Tetrahedra numbers\n");
	}
}



// class to hold the information at each node
class NodeArray{

private:
	int *MyTet;
	float *MyPos;
	float *MyA;
	float *MyForce;
	int size;


public:
	NodeArray(int l);
	~NodeArray();

	void set_pos(int i, int j,const float &newval);
	void set_tet(int i,const int &newval);
	void set_A(int i, int j, int k,const float &newval);
	void set_force(int i, int j,const float &newval);
	float get_pos(int i, int j);
	int get_tet(int i);
	float get_A(int i, int j, int k);
	float get_force(int i, int j);
	void switch_nodes(int i, int j);

};


NodeArray::NodeArray(int l){
	size = l;
	MyTet = new int[size];
	MyPos = new float[size*3];
	MyA = new float[size*3*3];
	MyForce = new float[size*3]; 
}

NodeArray::~NodeArray(){
	delete MyTet;
	MyTet = NULL;
	delete MyPos;
	MyPos = NULL;
	delete MyA;
	MyA = NULL;
	delete MyForce; 
	MyForce = NULL;
}

void NodeArray::set_pos(int i, int j, const float &newval){
	if(i<size && i>=0){
		if(j<3 && j>=0){
			MyPos[i*3+j] = newval;
			}else{
		printf("index out of bounds setting node position\n");
		}
	}else{
		printf("index out of bounds setting node position\n");
	}
}

void NodeArray::set_tet(int i, const int &newval){
	if(i<size && i>=0){
			MyTet[i] = newval;
	}else{
		printf("index out of bounds setting node to tet\n");
	}
}


void NodeArray::set_A(int i, int j, int k, const float &newval){
	if(i<size && i>=0){
		if(j<3 && j>=0){
			if(k<3 && k>=0){
			MyA[i*3*3+j*3+k] = newval;
			}else{
			printf("index out of bounds setting node A\n");
			}
			}else{
		printf("index out of bounds setting node A\n");
		}
	}else{
		printf("index out of bounds setting node A\n");
	}
}

void NodeArray::set_force(int i, int j, const float &newval){
	if(i<size && i>=0){
		if(j<3 && j>=0){
			MyForce[i*3+j] = newval;
			}else{
		printf("index out of bounds setting node force\n");
		}
	}else{
		printf("index out of bounds setting node force\n");
	}
}

float NodeArray::get_pos(int i, int j){
	if(i<size && i>=0){
		if(j<3 && j>=0){
			return MyPos[i*3+j];
			}else{
		printf("index out of bounds getting node position\n");
		return 1000.0;
		}
	}else{
		printf("index out of bounds getting node position\n");
		return 1000.0;
	}
}

int NodeArray::get_tet(int i){
	if(i<size && i>=0){
			return MyTet[i];
	}else{
		printf("index out of bounds getting node to tet\n");
		return -100;
	}
}


float NodeArray::get_A(int i, int j, int k){
	if(i<size && i>=0){
		if(j<3 && j>=0){
			if(k<3 && k>=0){
			return MyA[i*3*3+j*3+k];
			}else{
			printf("index out of bounds getting node A\n");
			return -100000;
			}
			}else{
		printf("index out of bounds getting node A\n");
		return -100000;
		}
	}else{
		printf("index out of bounds getting node A\n");
		return -100000;
	}
}

float NodeArray::get_force(int i, int j){
	if(i<size && i>=0){
		if(j<3 && j>=0){
			return MyForce[i*3+j];
			}else{
		printf("index out of bounds getting node force\n");
		return -10000.0;
		}
	}else{
		printf("index out of bounds getting node force\n");
		return -10000.0;
	}
}

void NodeArray::switch_nodes(int i, int j){

	float buffpos,bufftet,buffA,buffF; 

	if(i<size && i>=0){
		if(j<size && j>=0){
			for(int p = 0;p<9;p++){
				
				if(p<3){//switch force and position
					buffF = MyForce[i*3+p];
					MyForce[i*3+p] = MyForce[j*3+p];
					MyForce[j*3+p] = buffF;

					buffpos = MyPos[i*3+p];
					MyPos[i*3+p] = MyPos[j*3+p];
					MyPos[j*3+p] = buffpos;

					if(p<1){//switch mytet
					bufftet = MyTet[i];
					MyTet[i] = MyTet[j];
					MyTet[j] = bufftet;
					}
				}
				//switch A's
				buffA = MyA[i*9+p];
				MyA[i*9+p] = MyA[j*9+p];
				MyA[j*9+p] = buffA;
			}
		}else{
		printf("index out of bounds switching Node numbers\n");
		}
	}else{
		printf("index out of bounds switching Node numbers\n");
	}

}



#endif //__DATASTRUCT_H__
