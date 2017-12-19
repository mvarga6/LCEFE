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
#include "tri_array.h"
#include "element_array.h"
#include "parameters.h"

using namespace std;

struct MeshDimensions
{
	int Ntets, Nnodes, Ntris;
	real rmin[3], rmax[3];
};

static MeshDimensions get_gmsh_dim(string fileName)
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
	
	printf("\n[ INFO ] Nodes are %d-dimensional", node_dim);

	// Read elements (actually tets only)

	int type = 0; 
	int tet_num = 0;
	int tri_num = 0;
	level = 0;
	for( ; ; ){
		//printf("d\n", level);
		getline(input, text);
		if(input.eof()) break;

		if(level == 0)
		{
			if (text.compare(elemsStart) == 0) level = 1;
		}
		else if (level == 1)
		{
			s_to_i4(text, length, ierror); // get element_num
			level = 2;
		}
		else if (level == 2)
		{
			if (text.compare(elemsEnd) == 0) break;
			else 
			{
				s_to_i4(text, length, ierror); //read indx
				text.erase(0, length);

				type = s_to_i4(text, length, ierror); //read type
				text.erase(0, length);

				if (type == (int)TRIANGLE) tri_num++;
				if (type == (int)TETRAHEDRON) tet_num++; //count a tet
			}
		}
	}

	//Ntets = tet_num;
	//Nnodes = node_num;
	MeshDimensions dims;
	dims.Nnodes = node_num;
	dims.Ntets = tet_num;
	dims.Ntris = tri_num;
	printf("\n[ INFO ] %d nodes\n[ INFO ] %d tets\n", node_num, tet_num);
	input.close();
	return dims;
}

static MeshDimensions get_gmsh(string fileName, NodeArray &nodes, TetArray &tets, TriArray &tris, real MeshScale){
	
	
	MeshDimensions result;
	result.Nnodes = nodes.size;
	result.Ntets = tets.size;
	result.Ntris = tris.size;
	
	real min[3] = {999999.f, 999999.f, 999999.f};
	real max[3] = {-999999.f, -999999.f, -999999.f};
	ifstream input;
	int idx, c, k;
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
	if(!input)
	{
		cerr << "\n[*** CRITICAL ***] GMSH_SIZE_READ - Fatal error!\n";
		exit(1);
	}

	// Read nodes positions

	level = 0;
	for( ; ; )
	{
		getline(input, text);
		if(input.eof()) break;

		if (level == 0)
		{
			if (text.compare(nodesStart) == 0) level = 1;
		}
		else if (level == 1)
		{
			s_to_i4(text, length, ierror); // read node_num
			level = 2;
			idx = 0;
		}
		else if (level == 2)
		{
			if (text.compare(nodesEnd) == 0) break;
			else
			{
				s_to_i4(text, length, ierror); // read indx
				text.erase(0,length);
				for(c = 0; c < 3; c++)
				{
					x = s_to_r8(text, length, ierror);
					text.erase(0, length);
					nodes.set_pos(idx, c, x * MeshScale);
					
					// set max and mins
					if (x > max[c]) max[c] = x;
					if (x < min[c]) min[c] = x;
				}
				idx++; //next node
			}
		}
	}

	// Read elements (actually tets and triangle only)

	
	
	//
	// TODO: Use 'ntags' once TetArray and TriArray
	// are properly implementing MeshElementArray
	//
	int type, ntags, tet, tri;
	level = 0;
	for( ; ; )
	{
		getline(input, text);
		if (input.eof()) break;

		if (level == 0)
		{
			if (text.compare(elemsStart) == 0) level = 1;
		}
		else if (level == 1)
		{
			s_to_i4(text, length, ierror); // read element_num
			level = 2;
			tet = 0;
			tri = 0;
		}
		else if (level == 2)
		{
			if (text.compare(elemsEnd) == 0) break;
			else 
			{
				s_to_i4(text, length, ierror); //read idx
				text.erase(0, length);

				type = s_to_i4(text, length, ierror); //read type
				text.erase(0, length);

				// for (k = 0; k < 3; k++){ //read tags
				// 	s_to_i4(text, length, ierror); // read tag
				// 	text.erase(0, length);
				// }

				//
				// TODO: Use this block once TetArray and TriArray
				// are properly implementing MeshElementArray
				//
				ntags = s_to_i4(text, length, ierror); //read # of tags
				text.erase(0, length);
				
				int * tags = new int[ntags];
				for (int t = 0; t < ntags; t++) // read tags
				{
					tags[t] = s_to_i4(text, length, ierror);
					text.erase(0, length);
				}

				if (type == (int)TETRAHEDRON) //if it is a tet
				{ 
					for (k = 0; k < 4; k++)
					{ 
						// read nodes of tet
						idx = s_to_i4(text, length, ierror);
						text.erase(0, length);
						tets.set_nabs(tet, k, idx-1);
					}
					tet++; // next tet
				}
				else if (type == (int)TRIANGLE)
				{
					for (k = 0; k < 3; k++)
					{
						// read nodes of triangle
						idx = s_to_i4(text, length, ierror);
						text.erase(0, length);
						tris.set_node_idx(tri, k, idx-1);
					}
					tri++; // next tri
				}
				//
				// TODO: Use this block once TetArray and TriArray
				// are properly implementing MeshElementArray
				//
				// else if ((ElementType)type == TriTest->type)
				// {
				// 	int node_idx[3]; // container for node indices
				// 	for (int k = 0; k < 3; k++) // read node indices
				// 	{
				// 		node_idx[k] = s_to_i4(text, length, ierror);
				// 		text.erase(0, length);
				// 	}
				//
				// 	// add an element to array
				// 	TriTest->add_element(node_idx, ntags, tags);
				// 	delete[] tags; // clear tmp memory
				// 	tri++;
				// }
			}
		}
	}
	
	for(int d = 0; d < 3; d++)
	{
		result.rmax[d] = max[d];
		result.rmin[d] = min[d];
	}
	
	input.close();
	return result;
}


//set the positon of each array by averaging the positions
//of the nodes so we can arrange the tetrahedra in a smart
//order to optimize memmory calls in the GPU

static void get_tet_pos(NodeArray *Nodes, TetArray *Tets, real x_offset = 0.0f, real y_offset = 0.0f, real z_offset = 0.0f)
{
	int Ntets = Tets->size;
	int n0,n1,n2,n3;
	real xave,yave,zave;
	for (int i = 0; i < Ntets; i++)
	{
		n0 = Tets->get_nab(i,0);
		n1 = Tets->get_nab(i,1);
		n2 = Tets->get_nab(i,2);
		n3 = Tets->get_nab(i,3);

		xave = ((Nodes->get_pos(n0,0) \
			   +Nodes->get_pos(n1,0) \
			   +Nodes->get_pos(n2,0) \
			   +Nodes->get_pos(n3,0))/4.0)
			   - x_offset;

		yave = ((Nodes->get_pos(n0,1) \
			   +Nodes->get_pos(n1,1) \
			   +Nodes->get_pos(n2,1) \
			   +Nodes->get_pos(n3,1))/4.0)
			   - y_offset;

		zave = ((Nodes->get_pos(n0,2) \
			   +Nodes->get_pos(n1,2) \
			   +Nodes->get_pos(n2,2) \
			   +Nodes->get_pos(n3,2))/4.0)
			   - z_offset;

		Tets->set_pos(i,0,xave);
		Tets->set_pos(i,1,yave);
		Tets->set_pos(i,2,zave);
		Tets->set_pos(i,3,xave*xave+yave*yave+zave*zave);
	}
}

#endif
