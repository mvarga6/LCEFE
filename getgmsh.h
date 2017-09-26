#ifndef __GETGMSH_H__
#define __GETGMSH_H__
//=====================================

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cmath>
#include <cstring>
#include <string>
#include "extlib/gmsh_io/gmsh_io.hpp"
#include "classstruct.h"
#include "parameters.h"

using namespace std;

struct MeshDimensions
{
	int Ntets, Nnodes;
};

MeshDimensions get_gmsh_dim(string fileName)
{
	//int indx;
	bool ierror;
	ifstream input;
	//int k;
	int length;
	int level;
	const double r8_big = 1.0E+30;
	string text;
	string nodesStart("$Nodes");
	string nodesEnd("$EndNodes");
	string elemsStart("$Elements");
	string elemsEnd("$EndElements");
	double x, x_max, x_min;
	double y, y_max, y_min;
	double z, z_max, z_min;
	int node_num = 0;
	//int node_dim = 0;
	//int element_num = 0;
	//int element_dim = 0;

	x_max = -r8_big;
	x_min = +r8_big;
	y_max = -r8_big;
	y_min = +r8_big;
	z_max = -r8_big;
	z_min = +r8_big;

	input.open(fileName.c_str());

	if(!input){
		cerr << "\nGMSH_SIZE_READ - Fatal error!\n";
		exit(1);
	}

	// Read nodes positions

	level = 0;
	for( ; ; ){
		getline(input, text);
		if(input.eof()) break;

		if(level == 0){
			if(text.compare(nodesStart) == 0) level = 1;
		}
		else if(level == 1){
			node_num = s_to_i4(text, length, ierror); // get node_num
			level = 2;
		}
		else if(level == 2){
			if(text.compare(nodesEnd) == 0) break;
			else{
				s_to_i4(text, length, ierror); // read indx
				text.erase(0,length);

				x = s_to_r8(text, length, ierror);
				if(x > x_max) x_max = x;
				if(x < x_min) x_min = x;
				text.erase(0, length);

				y = s_to_r8(text, length, ierror);
				if(y > y_max) y_max = y;
				if(y > y_min) y_min = y;
				text.erase(0, length);

				z = s_to_r8(text, length, ierror);
				if(z > z_max) z_max = z;
				if(z > z_min) z_min = x;
				text.erase(0, length);
			}
		}
	}

	// Assume node dimensions
	int node_dim = 3;
	if(z_max == z_min){
		node_dim = 2;
		if(y_max == y_min){
			node_dim = 1;
		}
	}
	
	printf("\nNodes are %d-dimensional", node_dim);

	// Read elements (actually tets only)

	int type = 0; 
	int tet_num = 0;
	level = 0;
	for( ; ; ){
		//printf("d\n", level);
		getline(input, text);
		if(input.eof()) break;

		if(level == 0){
			if(text.compare(elemsStart) == 0) level = 1;
		}
		else if(level == 1){
			s_to_i4(text, length, ierror); // get element_num
			level = 2;
		}
		else if(level == 2){
			if(text.compare(elemsEnd) == 0) break;
			else {
				s_to_i4(text, length, ierror); //read indx
				text.erase(0, length);

				type = s_to_i4(text, length, ierror); //read type
				text.erase(0, length);
				//printf("id: %d\ttype: %d", indx, type);

				if(type == 4) tet_num++; //count a tet

				//k = 0;
				//for( ; ; k++){
				//	indx = stoi(text, &length);
				//	text.erase(0, length);
				//	if(text.size() == 0) break;
				//}
				//element_order = k - 3;
				//break;
			}

		}
	}

	//Ntets = tet_num;
	//Nnodes = node_num;
	MeshDimensions dims;
	dims.Nnodes = node_num;
	dims.Ntets = tet_num;
	printf("\n%d nodes\n%d tets\n", node_num, tet_num);
	input.close();
	return;
}

void get_gmsh(string fileName, NodeArray &nodes, TetArray &tets, float MeshScale){
	
	int Nnodes = nodes.size;
	int Ntets = tets.size;
	//int indx;
	ifstream input;
	int n, c, k;
	bool ierror;
	int length; //, igarb;
	int level;
	string text;
	string nodesStart("$Nodes");
	string nodesEnd("$EndNodes");
	string elemsStart("$Elements");
	string elemsEnd("$EndElements");
	double x;

	input.open(fileName.c_str());
	if(!input){
		cerr << "\nGMSH_SIZE_READ - Fatal error!\n";
		exit(1);
	}

	// Read nodes positions

	level = 0;
	for( ; ; ){
		getline(input, text);
		if(input.eof()) break;

		if(level == 0){
			if(text.compare(nodesStart) == 0) level = 1;
		}
		else if(level == 1){
			s_to_i4(text, length, ierror); // read node_num
			level = 2;
			n = 0;
		}
		else if(level == 2){
			if(text.compare(nodesEnd) == 0) break;
			else{
				s_to_i4(text, length, ierror); // read indx
				text.erase(0,length);
				for(c = 0; c < 3; c++){
					x = s_to_r8(text, length, ierror);
					text.erase(0, length);
					nodes.set_pos(n, c, x * MeshScale);
				}
				n++; //next node
			}
		}
	}

	// Read elements (actually tets only)
	
	int type, t;
	level = 0;
	for( ; ; ){
		getline(input, text);
		if(input.eof()) break;

		if(level == 0){
			if(text.compare(elemsStart) == 0) level = 1;
		}
		else if(level == 1){
			s_to_i4(text, length, ierror); // read element_num
			level = 2;
			t = 0;
		}
		else if(level == 2){
			if(text.compare(elemsEnd) == 0) break;
			else {
				s_to_i4(text, length, ierror); //read idx
				text.erase(0, length);

				type = s_to_i4(text, length, ierror); //read type
				text.erase(0, length);

				for(k = 0; k < 3; k++){ //read garbage
					s_to_i4(text, length, ierror); // read indx
					text.erase(0, length);
				}

				if(type == 4){ //if it is a tet
					for(k = 0; k < 4; k++){ //read nodes of tet
						n = s_to_i4(text, length, ierror);
						text.erase(0, length);
						tets.set_nabs(t, k, n-1);
					}
					t++; //next tet
				}
			}
		}
	}
	input.close();
	printf("\nNode positions and tetrahedral neighbors assigned.");
	return;
}


#endif
