#pragma once

#include "classstruct.h"
#include <fstream>
#include <sstream>
#include <vector>
#include "mainhead.h"

namespace mv {
	bool get_mesh(TetArray *tetArray, NodeArray *nodeArray, int Ntets, int Nnodes){
		
		//.. open and check open
		std::ifstream fin(MESHFILE, std::ios::in);
		if (!fin.is_open()) return false;

		//.. read in counts
		int garb, Ttot;
		fin >> garb >> Ttot;

		//.. read  data
		std::string line;
		float in;
		int line_count = 0, n0, n1, n2, n3;
		bool lines_read = false;
		while (std::getline(fin, line, '\n')){
			
			//.. grab line data
			std::stringstream ss(line);
			std::vector<float> line_data;
			while (ss >> in){
				line_data.push_back(in);
			}
			// ------------------------------------------------------------------
			// is a node
			const int size = line_data.size();
			if ((size == 4) && (++line_count <= Nnodes)){

				//.. set postions
				nodeArray->set_pos(int(line_data[0]), 
									line_data[1], 
									line_data[2], 
									line_data[3]);
			}
			// -------------------------------------------------------------------
			// is a line 
			else if ((size == 4) && (++line_count > Nnodes) && (!lines_read)){
				//.. do nothing
			}
			// -------------------------------------------------------------------
			// is a triangle 
			else if (size == 5){
				//.. flag that lines are done
				lines_read = true;
				++line_count;
			}
			// -------------------------------------------------------------------
			// is a tetra (at end of file)
			else if (size == 4 && lines_read && (++line_count >= (Ttot - Ntets))){
				//.. count of tets read
				static int tet = 0;

				//.. convert to ints, fix for c++ index, assign
				n0 = int(line_data[0]) - 1;
				n1 = int(line_data[1]) - 1;
				n2 = int(line_data[2]) - 1;
				n3 = int(line_data[3]) - 1;
				tetArray->set_nabs(tet++, n0, n1, n2, n3);
			}
			// ------------------------------------------------------------------
			// empty vector
			line_data.empty();
		}
		fin.close();
		return true;
	}
}