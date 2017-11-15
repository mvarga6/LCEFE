#include "mesh_operations.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
//#include "getmesh.h"
#include "genrand.h"
#include "getAs.h"
#include "helpers_math.h"

OperationResult SortOnTetrahedraPosition::Run(TetArray *Tets, NodeArray *Nodes, Logger *log)
{
	try
	{
		log->Msg("Running SortOnTetrahedraPosition ...");
		int Ntets = Tets->size;	
		
	
		
		log->Msg("Creating index list");
		// create value/key pairs to sort
		std::vector<std::pair<real, int> > posIdxList;
		
		// initialize the list
		for (int n = 0; n < Ntets; n++)
		{
			posIdxList.push_back(std::pair<real, int>(Tets->get_pos(n, 3), n));
		}
		
		// sort on position
		log->Msg("Sorting index list on distance from origin");
		std::sort(posIdxList.begin(), posIdxList.end());
		
		log->Msg("Copy new index list");
		std::vector<int> tetOrder(Ntets);
		for (int i = 0; i < Ntets; i++)
		{
			tetOrder.at(i) = posIdxList.at(i).second;
		}
		
		// put them into the proper order
		log->Msg("Reordering data");
		Tets->reorder(tetOrder);
				
		log->Msg("Completed!");
		
//		std::vector<int> tetIdx(Ntets);
//		std::iota(tetIdx.begin(), tetIdx.end(), 0);
//		
//		this->log->Msg("OLD WAY");
//		real dr1, dr2;
//		int count = 0;
//		int switched = 1;
//		stringstream ss;
//		while(switched > 0)
//		{
//			count++;
//			switched = 0;
//			
//			// loop at tets
//			for(int n1 = 0; n1 < Ntets - 1; n1++)
//			{
//				dr1 = Tets->get_pos(n1,3);
//				dr2 = Tets->get_pos(n1+1,3);
//				if (dr2 < dr1)
//				{
//					switched++;
//					Tets->switch_tets(n1,n1+1);
//				}
//			}//n1
//			
//			ss.str(std::string());
//			ss << "iteration = " << count << " reordered = " << switched;
//			this->log->Msg(ss.str());
//		}//go==1
//		ss.str(std::string());
//		
//		this->log->Msg("Completed!");
//		
//		this->log->Msg("Checking if they agree ...");
//		
//		// check that the first 1000 are correct
//		int correct = 0;
//		std::vector<std::pair<int,int>> incorrect;
//		for (int i = 0; i < 1000; i++)
//		{
//			if (tetIdx.at(i) == posIdxList.at(i).second) correct++;
//			else incorrect.push_back(std::pair<int,int>(tetIdx.at(i), posIdxList.at(i).second));
//		}
//		ss << correct << " of 1000 correct";
//		this->log->Msg(ss.str());
//		
//		// print first 10 incorrcect if there are some
//		if (incorrect.size() > 0)
//		{
//			for (int i = 0; i < incorrect.size(); i++)
//			{
//				if (i > 10) break;
//				
//				ss.str(std::string()); // clear
//				ss << i << ": " << incorrect.at(i).first << " " << incorrect.at(i).second;
//				this->log->Msg(ss.str());				
//			}
//		}
		
//		exit(1);
		
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// print something
		log->Msg("SortOnTetrahedraPosition threw exception -- quiting");
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}

MonteCarloMinimizeDistanceBetweenPairs::MonteCarloMinimizeDistanceBetweenPairs(const real kBTStart, const real kBTEnd, const real annealFactor)
{
	this->kbt_start = kBTStart;
	this->kbt_end = kBTEnd;
	this->anneal_factor = annealFactor;
}

OperationResult MonteCarloMinimizeDistanceBetweenPairs::Run(TetArray *Tets, NodeArray *Nodes, Logger *log)
{
	try
	{
		log->Msg("Running MonteCarloMinimizeDistanceBetweenPairs ...");
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
			if ((tot % 10) == 0)
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
		log->Msg("MonteCarloMinimizeDistanceBetweenPairs threw exception -- quiting");
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}


OperationResult ReassignIndices::Run(TetArray *Tets, NodeArray *Nodes, Logger *log)
{
	try
	{
		log->Msg("Running ReassignIndices ...");
		int Ntets = Tets->size;
		int Nnodes = Nodes->size;

		int nrank;
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
		int newi = 0;
		int i;
		for(int t = 0; t < Ntets; t++)
		{
			for (int tn = 0; tn < 4; tn++)
			{
				i = Tets->get_nab(t, tn);
				//printf("i = %d for t= %d and tn = %d\n",i,t,tn);
				if(Nodes->get_newnum(i) < 0)
				{
					Nodes->set_newnum(i, newi);
					newi++;
				}
			}
		}
		log->StaticMsg("Setting new node numbers: complete");

	
		//now reassign each tetrahedra to neighbors
		//in the new arrangement of nodes
		log->Msg("Assigning nodes to tetrahedra: ");
		for(int t = 0;t < Ntets; t++)
		{
			for (int tn = 0; tn < 4; tn++)
			{
				i = Tets->get_nab(t, tn);
				nrank = Nodes->get_totalRank(i);
				Tets->set_nabsRank(t, tn, nrank);
				Nodes->add_totalRank(i, 1);
				Tets->set_nabs(t, tn, Nodes->get_newnum(i));
			}
		}
		log->Msg("Assigning nodes to tetrahedra: complete");

		//switch actual order of nodes
		//do this by sweeping though and switching
		//nodes which have lower real val
		//not the most efficient sort but it will work
		log->Msg("Sorting nodes by value: ");
		bool go = true;
		while(go)
		{
			go = false;
			for(int i = 0; i < Nnodes-1; i++)
			{
				if (Nodes->get_newnum(i) > Nodes->get_newnum(i+1))
				{
					Nodes->switch_nodes(i,i+1);
					go = true;
				}
				if (Nodes->get_newnum(i) < 0)
				{
					//printf("nodes not properly reassigned node %d\n",i);
					stringstream ss;
					ss << "Nodes not properlyy reassigned node " << i;
					log->Msg(ss.str());
				}
			}
		}
		log->Msg("Sorting nodes by value: complete");
		
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// log something
		log->Msg("ReassignIndices threw exception -- quiting");
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}


SetDirector::SetDirector(DirectorField *director)
{
	this->director = director;
}


OperationResult SetDirector::Run(TetArray *Tets, NodeArray *Nodes, Logger *log)
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


OperationResult CalculateAinv::Run(TetArray *Tets, NodeArray *Nodes, Logger *log)
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


OperationResult CalculateVolumes::Run(TetArray *Tets, NodeArray *Nodes, Logger *log)
{
	try
	{
		log->Msg("Calculating tetrahedra volumes...");
	
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
			tempVol = tetVolume( Nodes->get_pos(n0,0)
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
	
	
		//calculate effective volume of each node
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

		//calculate total volume
		Tets->calc_total_volume();
		
		log->StaticMsg("Calculating tetrahedra volumes...\t\tcomplete");
		
		return OperationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		log->Error(e.what());
		return OperationResult::FAILURE_EXCEPTION_THROWN;
	}
}
