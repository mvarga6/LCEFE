#include "mesh_optimizer.h"
#include <sstream>
//#include "getmesh.h"
#include "genrand.h"

SortOnTetrahedraPosition::SortOnTetrahedraPosition(Logger *log)
{
	this->log = log;
}

OptimizationResult SortOnTetrahedraPosition::Run(TetArray *Tets, NodeArray *Nodes)
{
	try
	{
		this->log->Msg("Running SortOnTetrahedraPosition ...");
		int Ntets = Tets->size;	
		float dr1, dr2;
		int count = 0;
		int switched = 1;
		
		stringstream ss;
		
		while(switched > 0)
		{
			count++;
			switched = 0;
			
			// loop at tets
			for(int n1 = 0; n1 < Ntets - 1; n1++)
			{
				dr1 = Tets->get_pos(n1,3);
				dr2 = Tets->get_pos(n1+1,3);
				if (dr2 < dr1)
				{
					switched++;
					Tets->switch_tets(n1,n1+1);
				}
			}//n1
			
			ss.str(std::string());
			ss << "iteration = " << count << " reordered = " << switched;
			this->log->StaticMsg(ss.str());
		}//go==1
		
		this->log->Msg("Completed!");
		return OptimizationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// print something
		this->log->Msg("SortOnTetrahedraPosition threw exception -- quiting");
		return OptimizationResult::FAILURE_EXCEPTION_THROWN;
	}
}

MonteCarloMinimizeDistanceBetweenPairs::MonteCarloMinimizeDistanceBetweenPairs(const float kBTStart, const float kBTEnd, const float annealFactor, Logger *log)
{
	this->kbt_start = kBTStart;
	this->kbt_end = kBTEnd;
	this->anneal_factor = annealFactor;
	this->log = log;
}

OptimizationResult MonteCarloMinimizeDistanceBetweenPairs::Run(TetArray *Tets, NodeArray *Nodes)
{
	try
	{
		this->log->Msg("Running MonteCarloMinimizeDistanceBetweenPairs ...");
		int Ntets = Tets->size;	
		float olddist, newdist;
		int n1,n2;
		float KbT = this->kbt_start;
	    int count = 0;
		int tot = 0;
		int switched = 0;

		stringstream ss;
		ss << this->kbt_start << " < kbt < " << this->kbt_end << " annealFactor = " << this->anneal_factor;
		this->log->Msg(ss.str());
		
		//simple reordering scheme bassed only on spacial locallity
		while(KbT >= this->kbt_end)
		{
			tot++;
			count++;
			n1 = int(floor(genrand()*float(Ntets)));
			n2 = int(floor(genrand()*float(Ntets)));
		
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
		
		this->log->Msg("Completed!");
		return OptimizationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// log something
		this->log->Msg("MonteCarloMinimizeDistanceBetweenPairs threw exception -- quiting");
		return OptimizationResult::FAILURE_EXCEPTION_THROWN;
	}
}


ReassignIndices::ReassignIndices(Logger *log)
{
 this->log = log;
}

OptimizationResult ReassignIndices::Run(TetArray *Tets, NodeArray *Nodes)
{
	try
	{
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
		printf("init complete\n");
	
	
		//Loop though the lists of tetrahedra and if 
		//a node has not been reassigned reassign it
		//should keep nodes in same tetrahedra close 
		//in memory and should keep nodes which share 
		//tetrahedra also close in memory
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
		printf("Renumber complete newi = %d Nnodes=%d\n", newi, Nnodes);

	
		//now reassign each tetrahedra to neighbors
		//in the new arrangement of nodes
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
		printf("Reassign tets complete\n");

		//switch actual order of nodes
		//do this by sweeping though and switching
		//nodes which have lower real val
		//not the most efficient sort but it will work
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
					printf("nodes not properly reassigned node %d\n",i);
				}
			}
		}
		
		return OptimizationResult::SUCCESS;
	}
	catch (const std::exception& e)
	{
		// log something
		return OptimizationResult::FAILURE_EXCEPTION_THROWN;
	}
}

