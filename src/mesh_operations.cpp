#include "../include/mesh_operations.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
//#include "getmesh.h"
#include "../include/genrand.h"
#include "../include/getAs.h"
#include "../include/helpers_math.h"
#include "../include/getgmsh.h"

OperationResult SortOnTetrahedraPosition::Run(TetArray *Tets, NodeArray *Nodes, TriArray* Tris, Logger *log)
{
	try
	{
		log->Msg("Running Sort-On-Tetrahedra-Position ...");
		int Ntets = Tets->size;	
		
	
		
		log->Msg("Creating tetrahedra index list");
		// create value/key pairs to sort
		std::vector<std::pair<real, int> > posIdxList;
		
		// initialize the list
		for (int n = 0; n < Ntets; n++)
		{
			posIdxList.push_back(std::pair<real, int>(Tets->get_pos(n, 3), n));
		}
		
		// sort on position
		log->Msg("Sorting index list on tetrahedra CoM distance from origin");
		std::sort(posIdxList.begin(), posIdxList.end());
		
		log->Msg("Reordering data: ");
		std::vector<int> tetOrder(Ntets);
		for (int i = 0; i < Ntets; i++)
		{
			tetOrder.at(i) = posIdxList.at(i).second;
		}
		
		// put them into the proper order
		Tets->reorder(tetOrder);
				
		log->Msg("Reordering data: complete");
		
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// print something
		log->Msg("Sort-On-Tetrahedra-Position threw exception -- quiting");
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}

MonteCarloMinimizeDistanceBetweenPairs::MonteCarloMinimizeDistanceBetweenPairs(const real kBTStart, const real kBTEnd, const real annealFactor)
{
	this->kbt_start = kBTStart;
	this->kbt_end = kBTEnd;
	this->anneal_factor = annealFactor;
#ifdef __DEBUG__
	srand(0);
#else
	srand(time(NULL));    //seed random number generator
#endif
	mt_init();       //initialize random number generator
	purge();         //free up memory in random number generator
}

OperationResult MonteCarloMinimizeDistanceBetweenPairs::Run(TetArray *Tets, NodeArray *Nodes, TriArray* Tris, Logger *log)
{
	try
	{
		log->Msg("Running Monte-Carlo-Minimize-Distance-Between-Pairs ...");
		int Ntets = Tets->size;	
		real olddist, newdist;
		int n1,n2;
		real KbT = this->kbt_start;
	    int count = 0;
		int tot = 0;
		int switched = 0;

		stringstream ss;
		ss << this->kbt_start << " < kbt < " << this->kbt_end << " annealFactor = " << this->anneal_factor;
		log->Msg(ss.str());
		
		//simple reordering scheme bassed only on spacial locallity
		while(KbT >= this->kbt_end)
		{
			tot++;
			count++;
			n1 = int(floor(genrand()*real(Ntets)));
			n2 = int(floor(genrand()*real(Ntets)));
		
				olddist = Tets->dist(n1,n1+1) \
						+ Tets->dist(n1,n1-1) \
						+ Tets->dist(n2,n2+1) \
						+ Tets->dist(n2,n2-1);

				newdist = Tets->dist(n2,n1+1) \
						+ Tets->dist(n2,n1-1) \
						+ Tets->dist(n1,n2+1) \
						+ Tets->dist(n1,n2-1);

				if (newdist < olddist)
				{
					Tets->switch_tets(n1,n2);
					count = 0;
					switched++;
				}
				else if (genrand() < exp(-(newdist-olddist)/(KbT)))
				{
					Tets->switch_tets(n1,n2);
					count = 0;
					switched++;
				}	

			KbT *= this->anneal_factor; //KbT*0.9999999;
			if ((tot % 1000) == 0)
			{
				//printf("KbT = %f count = %d\n",KbT,count);
				ss.str(std::string());
				ss << "kBT = " << KbT << "\treordered = " << switched;
				log->StaticMsg(ss.str());
				switched = 0;
			}
		}
		
		log->Msg("Completed!");
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// log something
		log->Msg("Monte-Carlo-Minimize-Distance-Between-Pairs threw exception -- quiting");
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}


OperationResult ReassignIndices::Run(TetArray *Tets, NodeArray *Nodes, TriArray* Tris, Logger *log)
{
	try
	{
		log->Msg("Running Reassign-Indices ...");
		int Ntets = Tets->size;
		int Nnodes = Nodes->size;
		int Ntris = Tris->size;

		//set all new node numbers negative so we can 
		//see when one is replaced and not replace it again
		//this should account for all the redundancies
		//in the tet nab lists
		for(int i = 0; i < Nnodes; i++)
		{
			Nodes->set_newnum(i,-100);
		}
	
	
		//Loop though the lists of tetrahedra and if 
		//a node has not been reassigned reassign it
		//should keep nodes in same tetrahedra close 
		//in memory and should keep nodes which share 
		//tetrahedra also close in memory
		log->Msg("Setting new node numbers: ");
		int new_idx = 0;
		int node_idx;
		int node_rank;
		for(int t = 0; t < Ntets; t++)
		{
			for (int tn = 0; tn < 4; tn++)
			{
				node_idx = Tets->get_nab(t, tn);
				if (Nodes->get_newnum(node_idx) < 0)
				{
					Nodes->set_newnum(node_idx, new_idx);
					new_idx++;
				}
			}
		}
		log->StaticMsg("Setting new node numbers: complete");

	
		// now reassign each tetrahedra to neighbors
		// in the new arrangement of nodes
		log->Msg("Assigning new node config to tetrahedra: ");
		std::vector<int> newOrder(Nnodes);
		for(int t = 0; t < Ntets; t++)
		{
			for (int n = 0; n < 4; n++)
			{
				// get node_idx for tn-th node in tet t
				node_idx = Tets->get_nab(t, n);

				// get rank of this node wrt tetrhedra
				node_rank = Nodes->get_rank_wrt_tets(node_idx);

				// set rank in tet array
				Tets->set_nabsRank(t, n, node_rank);

				// increment the rank of node wrt to tets
				Nodes->increment_rank_wrt_tets(node_idx);

				// set the node_idx for the n-th node of
				// the t-th tet as the new node number
				int new_idx = Nodes->get_newnum(node_idx);
				Tets->set_nabs(t, n, new_idx);

				// record the new node_idx for reordering data later
				newOrder.at(node_idx) = new_idx;
			}
		}
		log->StaticMsg("Assigning new node config to tetrahedra: complete");

		// reassign each triangle to neighbors
		// in the new arrangement of nodes
		// TODO: Check that this is assigning working
		log->Msg("Assigning new node config to triangles: ");
		for (int t = 0; t < Ntris; t++)
		{
			for (int n = 0; n < 3; n++)
			{
				// get the node index
				node_idx = Tris->node_idx(t, n);

				// get the rank of this node wrt triangles
				node_rank = Nodes->get_rank_wrt_tris(node_idx);

				// set the rank of n-th node in t-th triangle
				Tris->set_rank(t, n, node_rank);

				// increment the rank of node wrt to triangles
				Nodes->increment_rank_wrt_tris(node_idx);

				// set the node_idx for the n-th node of
				// the t-th tri as the new node number
				Tris->set_node_idx(t, n, Nodes->get_newnum(node_idx));
			}
		}
		log->StaticMsg("Assigning new node config to triangles: complete");

		///
		/// Now actually reorder the node data into the new order
		///

		log->Msg("Reordering nodes in this configuration: ");
		Nodes->reorder(newOrder);
		log->StaticMsg("Reordering nodes in this configuration: complete");
		
		int outOfOrderCount = 0;
		for (int i = 0; i < Nnodes; i++)
		{
			if (i != Nodes->get_newnum(i)) 
			{
				outOfOrderCount++;
			}
		}

		if (outOfOrderCount > 0)
		{
			printf("\n[ ERROR ] # of nodes out of order: %d", outOfOrderCount);
		}
		

		//switch actual order of nodes
		//do this by sweeping though and switching
		//nodes which have lower real val
		//not the most efficient sort but it will work
		// log->Msg("Sorting nodes by value: ");
		// bool go = false;
		// while(go)
		// {
		// 	go = false;
		// 	for(int i = 0; i < Nnodes-1; i++)
		// 	{
		// 		if (Nodes->get_newnum(i) > Nodes->get_newnum(i+1))
		// 		{
		// 			Nodes->switch_nodes(i,i+1);
		// 			go = true;
		// 		}
		// 		if (Nodes->get_newnum(i) < 0)
		// 		{
		// 			//printf("nodes not properly reassigned node %d\n",i);
		// 			stringstream ss;
		// 			ss << "Nodes not properlyy reassigned node " << i;
		// 			log->Msg(ss.str());
		// 		}
		// 	}
		// }
		// log->StaticMsg("Sorting nodes by value: complete");
		
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// log something
		log->Msg("Reassign-Indices threw exception -- quiting");
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}


SetDirector::SetDirector(DirectorField *director)
{
	this->director = director;
}


OperationResult SetDirector::Run(TetArray *Tets, NodeArray *Nodes, TriArray* Tris, Logger *log)
{
	try
	{
		log->Msg("Assigning director to mesh...");
	
		int Ntets = Tets->size;
	
		DirectorOrientation dir;
		real x, y, z;
	
		for (int t = 0; t < Ntets; t++)
		{
			// the position of this tet
			x = Tets->get_pos(t, 0);
			y = Tets->get_pos(t, 1);
			z = Tets->get_pos(t, 2);
	
			// get the director there
			// done this way so it could be read
			// from file or hardcoded etc
			dir = this->director->GetDirectorAt(x, y, z);
			
			// assign 
			Tets->set_theta(t, dir.theta);
			Tets->set_phi(t, dir.phi);
		}
		
		log->StaticMsg("Assigning director to mesh...\t\tcomplete");
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// log something
		log->Msg("SetDirector threw exception -- quiting");
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}


OperationResult CalculateAinv::Run(TetArray *Tets, NodeArray *Nodes, TriArray* Tris, Logger *log)
{
	try
	{
		log->Msg("Calculating tetrahedra shape functions...");
		init_As(*Nodes, *Tets);
		log->StaticMsg("Calculating tetrahedra shape functions...\tcomplete");
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		log->Error(e.what());
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}


OperationResult CalculateProperties::Run(TetArray *Tets, NodeArray *Nodes, TriArray* Tris, Logger *log)
{
	try
	{
		log->Msg("Calculating tetrahedra volumes: ");
	
		real tempVol;
		int n0, n1, n2, n3;
		int Ntets = Tets->size;
	
		//calculate volume of each tetrahedra
		for(int t = 0;t < Ntets; t++)
		{
			n0 = Tets->get_nab(t,0);
			n1 = Tets->get_nab(t,1);
			n2 = Tets->get_nab(t,2);
			n3 = Tets->get_nab(t,3);
			tempVol = math::tetVolume( 
						Nodes->get_pos(n0,0)
						,Nodes->get_pos(n0,1)
						,Nodes->get_pos(n0,2)
						,Nodes->get_pos(n1,0)
						,Nodes->get_pos(n1,1)
						,Nodes->get_pos(n1,2)
						,Nodes->get_pos(n2,0)
						,Nodes->get_pos(n2,1)
						,Nodes->get_pos(n2,2)
						,Nodes->get_pos(n3,0)
						,Nodes->get_pos(n3,1)
						,Nodes->get_pos(n3,2));

			Tets->set_volume(t,tempVol);
		}

		//calculate total volume
		Tets->calc_total_volume();

		log->StaticMsg("Calculating tetrahedra volumes: complete");

		///
		/// Triangle property calculations
		///
		log->Msg("Calculating surface triangles' areas: ");
		Tris->update_areas(Nodes);
		log->StaticMsg("alculating surface triangles' areas: complete");

		log->Msg("alculating surface triangles' normals: ");
		Tris->update_normals(Nodes, Nodes->centroid());
		log->StaticMsg("alculating surface triangles' normals: complete");

		/// calculate effective volume of each node
		int i;
		for(int t = 0; t < Ntets; t++)
		{
			tempVol = 0.25 * Tets->get_volume(t);
			for (int tn = 0; tn < 4; tn++)
			{
				i = Tets->get_nab(t,tn);
				Nodes->add_volume(i,tempVol);
			}
		}

		//normalize volume so that each node
		//has an average volume of 1
		//i_Node.normalize_volume(real(Nnodes));
		

		log->Msg("Calculating tetrahedra shape functions: ");
		init_As(*Nodes, *Tets);
		log->StaticMsg("Calculating tetrahedra shape functions: complete");

		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		log->Error(e.what());
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}


EulerRotation::EulerRotation(const real theta, const real phi, const real rho)
{
	this->theta = theta;
	this->phi = phi;
	this->rho = rho;
}


OperationResult EulerRotation::Run(TetArray *Tets, NodeArray *Nodes, TriArray* Tris, Logger *log)
{
	log->Msg("Rotation nodes of mesh.");

	// rotate nodes
	Nodes->eulerRotation(theta, phi, rho);

	log->Msg("Recalculating tetrahedra positions");

	// recalculate tet positions
	get_tet_pos(Nodes, Tets);

	log->Msg("Complete");
}